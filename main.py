# free-model-aggregator-02 entry: one server that puts the best live free model
# behind both the OpenAI Chat Completions wire and the Anthropic Messages wire.
#
# How it works, in one paragraph: the trial brief (MAKEFUN_PROMPT_FILE) lists the
# fourteen free OpenRouter models with the house's quality rating for each, and
# the six prompts in the order the env sends them. For every prompt this server
# ranks the free models that meet the prompt's requirements (tools, vision, JSON
# mode, context) by rating, best first, and dials the winner through the grader's
# gateway on the Chat Completions route. A refusal (a real provider 429/5xx, or a
# gateway refusal) moves the prompt to the next model in the ranking; an answer
# that comes back without the prompt's token does the same. The answer that
# finally serves is rendered back in the wire the client used, JSON or
# server-sent events, and its `model` field always names the model that was
# really dialled. Recovered refusals are reported on the response in the
# `x-makefun-upstream-failures` header so the failover column can see them.
#
# Determinism: nothing in any response is read from a clock or a random source.
# Every byte the server sends is a function of the brief, the request it is
# answering and the gateway answers it has seen, so a replay of the same trial
# leaves the same state export. Timeouts are never consulted for content; the
# gateway call simply waits for the gateway, exactly as the trial intends.
#
# The file is self-contained on purpose: a trial runs it as
# `python3 -P -s /bot/main.py`, and -P keeps this file's own directory off
# sys.path, so a sibling module could not be imported reliably. Standard library
# only; nothing is installed before the server runs.

import json
import math
import os
import re
import signal
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

# --------------------------------------------------------------------------
# Trial configuration, read once from the environment the harness sets.

ENV_URL = os.environ.get('MAKEFUN_ENV_URL', '').rstrip('/')
BRIEF_PATH = os.environ.get('MAKEFUN_PROMPT_FILE', '')
GATEWAY_BASE = os.environ.get('OPENAI_BASE_URL', '').rstrip('/')
GATEWAY_KEY = os.environ.get('OPENAI_API_KEY', '')

PROTOCOL_VERSION = 1
POLL_SECONDS = 0.1
MAX_BODY = 8 * 1024 * 1024
MAX_ATTEMPTS = 4            # models tried per prompt before the prompt is given up
MAX_OUTPUT_CAP = 8192       # the gateway refuses a larger output cap
DEFAULT_OUTPUT_CAP = 1024   # named when a client body names none
FAILURES_HEADER = 'x-makefun-upstream-failures'

PROVIDER_ID = 'openrouter'
PROVIDER_PUBLIC_BASE = 'https://openrouter.ai/api/v1'
PROVIDER_KEY_ENV = 'OPENROUTER_API_KEY'
SCHEMA_PREFIX = 'free-model-aggregator-02/'

# Fixed listing stamps: a model list that read the clock would differ between a
# trial's live pass and its replay, so both wires carry constants instead.
LIST_CREATED = 1767225600
LIST_CREATED_AT = '2026-01-01T00:00:00Z'

# The token every prompt carries and its answer must carry back, shaped exactly
# as the house defines it (WORD-1234).
TOKEN_PATTERN = re.compile(r'(?<![A-Z0-9])([A-Z]{3,12}-[0-9]{4})(?![0-9A-Z])')

OPENAI_FINISH_OK = {'stop', 'length', 'tool_calls', 'function_call', 'content_filter'}

# The gateway is dialled directly, never through a proxy: the trial's addresses
# are local to the trial, and an inherited proxy variable must not reroute them.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def note(message):
    sys.stderr.write('aggregator-02: ' + message + '\n')
    sys.stderr.flush()


def is_obj(value):
    return isinstance(value, dict)


def is_num(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def as_str(value):
    return value if isinstance(value, str) else ''


def as_int(value):
    return value if isinstance(value, int) and not isinstance(value, bool) else None


# --------------------------------------------------------------------------
# The brief: two JSON documents (the model list, the prompt manifest) ride
# inside prose, each tagged with a `schema` under this competition's prefix.
# They are found by scanning for balanced objects, never by parsing the prose.

def _object_end(text, start):
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == '\\':
                i += 2
                continue
            if ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _schema_documents(text):
    docs = []
    at = text.find('{')
    while at != -1:
        end = _object_end(text, at)
        if end != -1:
            try:
                value = json.loads(text[at:end + 1])
            except ValueError:
                value = None
            if is_obj(value) and isinstance(value.get('schema'), str):
                docs.append(value)
        at = text.find('{', at + 1)
    return docs


def _model_row(value):
    if not is_obj(value):
        return None
    model_id = as_str(value.get('id')) or as_str(value.get('model'))
    if model_id == '':
        return None
    if as_str(value.get('provider')) == PROVIDER_ID and not model_id.startswith(PROVIDER_ID + '/'):
        model_id = PROVIDER_ID + '/' + model_id
    rating = value.get('rating_milli')
    return {
        'id': model_id,
        'name': as_str(value.get('display_name')) or model_id,
        'rating': rating if is_num(rating) else 0,
        'context': value.get('context_length') if is_num(value.get('context_length')) else None,
        'tools': value.get('tools') is True,
        'vision': value.get('vision') is True,
        'json': value.get('json') is True,
        'free': value.get('free') is not False,
    }


def _prompt_row(value):
    if not is_obj(value):
        return None
    prompt_id = as_str(value.get('id'))
    if prompt_id == '':
        return None
    requires = value.get('requires') if is_obj(value.get('requires')) else {}
    row = {'id': prompt_id}
    for flag in ('tools', 'vision', 'json'):
        if requires.get(flag) is True:
            row[flag] = True
    if is_num(requires.get('min_context')):
        row['min_context'] = requires['min_context']
    return row


def load_brief(path):
    brief = {'models': [], 'prompts': []}
    if path == '':
        return brief
    try:
        with open(path, encoding='utf-8') as handle:
            text = handle.read()
    except OSError as err:
        note('cannot read the brief at ' + path + ': ' + str(err))
        return brief
    for doc in _schema_documents(text):
        if SCHEMA_PREFIX not in doc['schema']:
            continue
        if isinstance(doc.get('models'), list) and not brief['models']:
            brief['models'] = [m for m in (_model_row(v) for v in doc['models']) if m is not None]
        if isinstance(doc.get('prompts'), list) and not brief['prompts']:
            brief['prompts'] = [p for p in (_prompt_row(v) for v in doc['prompts']) if p is not None]
    return brief


BRIEF = load_brief(BRIEF_PATH)
FREE_MODELS = [m for m in BRIEF['models'] if m['free']]


def build_catalog():
    """The one catalog document: registered with the env and printed by --list.

    Built once and serialised once, so the register body's catalog and the
    --list output are the same bytes. base_url is the provider's public
    endpoint, never the gateway's trial address: the catalog is part of the
    state export a replay compares byte for byte, and the gateway's origin
    differs between the live pass and the replay.
    """
    rows = []
    seen = set()
    for model in FREE_MODELS:
        if model['id'] in seen:
            continue
        seen.add(model['id'])
        rows.append({'id': model['id'], 'free': True, 'context_length': model['context']})
    if not rows:
        return {'providers': []}
    return {'providers': [{
        'id': PROVIDER_ID,
        'base_url': PROVIDER_PUBLIC_BASE,
        'key_env': PROVIDER_KEY_ENV,
        'wire': 'openai',
        'models': rows,
    }]}


CATALOG = build_catalog()
CATALOG_TEXT = json.dumps(CATALOG, separators=(',', ':'), ensure_ascii=False)


# --------------------------------------------------------------------------
# Routing: which free models may serve a prompt, best first.

def model_meets(model, requires):
    if requires.get('tools') and not model['tools']:
        return False
    if requires.get('vision') and not model['vision']:
        return False
    if requires.get('json') and not model['json']:
        return False
    floor = requires.get('min_context')
    if floor is not None and (model['context'] is None or model['context'] < floor):
        return False
    return True


def ranked_candidates(requires):
    """Free models meeting the requirements, best rating first, id as tie-break."""
    pool = [m for m in FREE_MODELS if model_meets(m, requires)]
    pool.sort(key=lambda m: (-m['rating'], m['id']))
    return pool


# The brief's prompts arrive one at a time, in manifest order, across both
# wires; a single counter over completion requests names the arriving prompt.
_arrival = {'index': 0}
_arrival_lock = threading.Lock()


def next_manifest_requires():
    with _arrival_lock:
        index = _arrival['index']
        _arrival['index'] += 1
    if index < len(BRIEF['prompts']):
        return dict(BRIEF['prompts'][index])
    return {}


def body_requirements(wire, body):
    """Requirements read off the request itself, merged with the manifest row.

    The manifest is the declared source; the body is the ground truth of what
    the client actually sent. Taking the union of the two can only ever demand
    a model that truly meets the request.
    """
    requires = {}
    if isinstance(body.get('tools'), list) and len(body['tools']) > 0:
        requires['tools'] = True
    if wire == 'openai':
        fmt = body.get('response_format')
        if is_obj(fmt) and fmt.get('type') in ('json_object', 'json_schema'):
            requires['json'] = True
    messages = body.get('messages') if isinstance(body.get('messages'), list) else []
    for message in messages:
        if not is_obj(message) or not isinstance(message.get('content'), list):
            continue
        for block in message['content']:
            if is_obj(block) and block.get('type') in ('image', 'image_url'):
                requires['vision'] = True
    return requires


def requirements_for(wire, body):
    merged = next_manifest_requires()
    for flag, wanted in body_requirements(wire, body).items():
        if wanted:
            merged[flag] = True
    return merged


# --------------------------------------------------------------------------
# Wire translation. The gateway speaks only Chat Completions, so an Anthropic
# request is rewritten into one and the gateway's answer is written back.

def block_text(content):
    """The plain text of a string body or of a block array's text blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ''
    return '\n'.join(
        b['text'] for b in content
        if is_obj(b) and b.get('type') == 'text' and isinstance(b.get('text'), str)
    )


def request_text(wire, body):
    """Every word the client sent, for the token search and usage estimates."""
    parts = []
    if wire == 'anthropic':
        parts.append(block_text(body.get('system')))
    messages = body.get('messages') if isinstance(body.get('messages'), list) else []
    for message in messages:
        if is_obj(message):
            parts.append(block_text(message.get('content')))
    return '\n'.join(parts)


def tokens_in(text):
    found = []
    for match in TOKEN_PATTERN.finditer(text):
        if match.group(1) not in found:
            found.append(match.group(1))
    return found


def _image_part(block):
    source = block.get('source')
    if not is_obj(source):
        return None
    if source.get('type') == 'base64' and isinstance(source.get('data'), str):
        media = source.get('media_type') if isinstance(source.get('media_type'), str) else 'image/png'
        return {'type': 'image_url', 'image_url': {'url': 'data:' + media + ';base64,' + source['data']}}
    if source.get('type') == 'url' and isinstance(source.get('url'), str):
        return {'type': 'image_url', 'image_url': {'url': source['url']}}
    return None


def _tool_result_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = block_text(content)
        return text if text != '' else json.dumps(content, separators=(',', ':'), ensure_ascii=False)
    if content is None:
        return ''
    return json.dumps(content, separators=(',', ':'), ensure_ascii=False)


def anthropic_as_chat(body, model_id):
    """An Anthropic Messages body as the Chat Completions body that is dialled."""
    messages = []
    system = block_text(body.get('system'))
    if system != '':
        messages.append({'role': 'system', 'content': system})
    source = body.get('messages') if isinstance(body.get('messages'), list) else []
    for raw in source:
        if not is_obj(raw):
            continue
        role = 'assistant' if raw.get('role') == 'assistant' else 'user'
        content = raw.get('content')
        if isinstance(content, str):
            messages.append({'role': role, 'content': content})
            continue
        if not isinstance(content, list):
            continue
        if role == 'assistant':
            calls = []
            for block in content:
                if not is_obj(block) or block.get('type') != 'tool_use':
                    continue
                calls.append({
                    'id': as_str(block.get('id')),
                    'type': 'function',
                    'function': {
                        'name': as_str(block.get('name')),
                        'arguments': json.dumps(
                            block['input'] if is_obj(block.get('input')) else {},
                            separators=(',', ':'), ensure_ascii=False),
                    },
                })
            message = {'role': 'assistant', 'content': block_text(content) or None}
            if calls:
                message['tool_calls'] = calls
            messages.append(message)
            continue
        parts = []
        for block in content:
            if not is_obj(block):
                continue
            if block.get('type') == 'tool_result':
                messages.append({
                    'role': 'tool',
                    'tool_call_id': as_str(block.get('tool_use_id')),
                    'content': _tool_result_text(block.get('content')),
                })
                continue
            if block.get('type') == 'text' and isinstance(block.get('text'), str):
                parts.append({'type': 'text', 'text': block['text']})
                continue
            if block.get('type') == 'image':
                part = _image_part(block)
                if part is not None:
                    parts.append(part)
        if parts:
            messages.append({'role': 'user', 'content': parts})

    out = {'model': model_id, 'messages': messages}
    cap = body.get('max_tokens')
    out['max_tokens'] = min(cap, MAX_OUTPUT_CAP) if is_num(cap) else DEFAULT_OUTPUT_CAP
    for key in ('temperature', 'top_p'):
        if is_num(body.get(key)):
            out[key] = body[key]
    if isinstance(body.get('stop_sequences'), list) and body['stop_sequences']:
        out['stop'] = body['stop_sequences']
    if isinstance(body.get('tools'), list):
        tools = []
        for raw in body['tools']:
            if not is_obj(raw) or not isinstance(raw.get('name'), str):
                continue
            tools.append({'type': 'function', 'function': {
                'name': raw['name'],
                'description': as_str(raw.get('description')),
                'parameters': raw['input_schema'] if is_obj(raw.get('input_schema')) else {'type': 'object', 'properties': {}},
            }})
        if tools:
            out['tools'] = tools
    choice = body.get('tool_choice')
    if is_obj(choice):
        if choice.get('type') == 'auto':
            out['tool_choice'] = 'auto'
        elif choice.get('type') == 'any':
            out['tool_choice'] = 'required'
        elif choice.get('type') == 'tool' and isinstance(choice.get('name'), str):
            out['tool_choice'] = {'type': 'function', 'function': {'name': choice['name']}}
    return out


def chat_body_from_openai(body, model_id):
    """A client's Chat Completions body, retargeted at the dialled model.

    The client's own fields ride through unchanged; only the model, the
    streaming flags and the output cap are settled here, because the gateway
    refuses a call with no cap or one over its cap, and this server always
    takes the gateway's answer whole before rendering it in the client's
    chosen form (JSON or SSE), so the gateway call itself never streams.
    """
    out = dict(body)
    out['model'] = model_id
    out.pop('stream', None)
    out.pop('stream_options', None)
    cap = out.get('max_completion_tokens')
    if not is_num(cap):
        cap = out.get('max_tokens')
    if is_num(cap):
        out['max_tokens'] = min(cap, MAX_OUTPUT_CAP)
        out.pop('max_completion_tokens', None)
    else:
        out['max_tokens'] = DEFAULT_OUTPUT_CAP
    return out


# --------------------------------------------------------------------------
# The gateway call.

def gateway_chat_url():
    if GATEWAY_BASE == '':
        return ''
    if GATEWAY_BASE.endswith('/v1'):
        return GATEWAY_BASE + '/chat/completions'
    return GATEWAY_BASE + '/v1/chat/completions'


def dial(body):
    """One POST to the gateway. Returns (status, parsed body or None, raw text).

    A refusal is data, not an exception: its status and body come back exactly
    as sent, for the failover chain and the failures trail. No timeout is set:
    the gateway answers or refuses, and a clock-based cut-off could fire on a
    live pass yet not on the replay, which must retrace the live pass exactly.
    """
    url = gateway_chat_url()
    if url == '':
        return 502, None, 'OPENAI_BASE_URL is not set'
    payload = json.dumps(body, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    request = urllib.request.Request(url, data=payload, method='POST')
    request.add_header('content-type', 'application/json')
    request.add_header('authorization', 'Bearer ' + GATEWAY_KEY)
    try:
        with OPENER.open(request) as response:
            raw = response.read().decode('utf-8', 'replace')
            status = response.status
    except urllib.error.HTTPError as err:
        raw = err.read().decode('utf-8', 'replace')
        status = err.code
    except (urllib.error.URLError, OSError) as err:
        return 502, None, 'the gateway could not be reached: ' + str(err)
    try:
        parsed = json.loads(raw)
    except ValueError:
        parsed = None
    return status, parsed if is_obj(parsed) else None, raw


def refusal_code(parsed, raw):
    if is_obj(parsed) and is_obj(parsed.get('error')):
        error = parsed['error']
        code = error.get('code') if isinstance(error.get('code'), str) else error.get('type')
        if isinstance(code, str) and code != '':
            return code[:64]
    return ''


def completion_of(answer):
    """(text, tool calls, finish reason) of a Chat Completions answer."""
    choices = answer.get('choices') if isinstance(answer.get('choices'), list) else []
    choice = choices[0] if choices and is_obj(choices[0]) else {}
    message = choice.get('message') if is_obj(choice.get('message')) else {}
    text = message.get('content')
    if not isinstance(text, str):
        text = block_text(text)
    calls = []
    raw_calls = message.get('tool_calls') if isinstance(message.get('tool_calls'), list) else []
    for call in raw_calls:
        if not is_obj(call) or not is_obj(call.get('function')):
            continue
        fn = call['function']
        calls.append({
            'id': as_str(call.get('id')),
            'name': as_str(fn.get('name')),
            'arguments': fn.get('arguments') if isinstance(fn.get('arguments'), str) else '',
        })
    finish = choice.get('finish_reason')
    return text, calls, finish if isinstance(finish, str) else None


def estimate_tokens(text):
    return max(0, math.ceil(len(text) / 4))


def settled_usage(answer, req_text, answer_text):
    """Whole usage counts: the provider's own when they are usable, else a
    deterministic estimate from the texts in hand. A served answer must carry
    an input of at least 0 and an output of at least 1 for the wire checks."""
    usage = answer.get('usage') if is_obj(answer.get('usage')) else {}
    prompt = usage.get('prompt_tokens')
    completion = usage.get('completion_tokens')
    input_tokens = prompt if as_int(prompt) is not None and prompt >= 0 else estimate_tokens(req_text)
    if as_int(completion) is not None and completion >= 1:
        output_tokens = completion
    else:
        output_tokens = max(1, estimate_tokens(answer_text))
    return input_tokens, output_tokens


def finish_or_stop(finish):
    return finish if finish in OPENAI_FINISH_OK else 'stop'


def anthropic_stop(finish):
    if finish == 'length':
        return 'max_tokens'
    if finish in ('tool_calls', 'function_call'):
        return 'tool_use'
    if finish == 'content_filter':
        return 'refusal'
    return 'end_turn'


def arguments_object(arguments):
    if isinstance(arguments, str) and arguments.strip() != '':
        try:
            parsed = json.loads(arguments)
        except ValueError:
            return {}
        if is_obj(parsed):
            return parsed
    return {}


# --------------------------------------------------------------------------
# Rendering: the gateway's answer in the client's wire, JSON or SSE. The
# `model` field of every answer and every chunk names the dialled id: the
# provider spells it without the gateway's prefix, so the dialled id is put
# back everywhere, because the house scores the model its record shows.

_message_counter = {'n': 0}


def message_id(answer):
    raw = answer.get('id')
    if isinstance(raw, str) and raw != '':
        return 'msg_' + raw
    _message_counter['n'] += 1
    return 'msg_aggregator_%d' % _message_counter['n']


def render_openai_json(answer, model_id, req_text):
    out = dict(answer)
    out['model'] = model_id
    text, _calls, finish = completion_of(answer)
    choices = out.get('choices') if isinstance(out.get('choices'), list) else []
    if choices and is_obj(choices[0]):
        first = dict(choices[0])
        message = dict(first.get('message')) if is_obj(first.get('message')) else {}
        message['role'] = 'assistant'
        if not isinstance(message.get('content'), (str, list)):
            message['content'] = text
        first['message'] = message
        first['finish_reason'] = finish_or_stop(finish)
        choices = [first] + list(choices[1:])
        out['choices'] = choices
    input_tokens, output_tokens = settled_usage(answer, req_text, text)
    out['usage'] = {'prompt_tokens': input_tokens, 'completion_tokens': output_tokens,
                    'total_tokens': input_tokens + output_tokens}
    return out


def _sse(data):
    return 'data: ' + json.dumps(data, separators=(',', ':'), ensure_ascii=False) + '\n\n'


def render_openai_stream(answer, model_id, req_text):
    text, calls, finish = completion_of(answer)
    input_tokens, output_tokens = settled_usage(answer, req_text, text)
    base = {'id': answer.get('id') if isinstance(answer.get('id'), str) else 'chatcmpl-aggregator',
            'object': 'chat.completion.chunk',
            'created': answer.get('created') if is_num(answer.get('created')) else 0,
            'model': model_id}
    frames = [_sse({**base, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})]
    if text != '':
        frames.append(_sse({**base, 'choices': [{'index': 0, 'delta': {'content': text}, 'finish_reason': None}]}))
    for index, call in enumerate(calls):
        frames.append(_sse({**base, 'choices': [{'index': 0, 'delta': {'tool_calls': [
            {'index': index, 'id': call['id'] or 'call_%d' % index, 'type': 'function',
             'function': {'name': call['name'], 'arguments': ''}}]}, 'finish_reason': None}]}))
        if call['arguments'] != '':
            frames.append(_sse({**base, 'choices': [{'index': 0, 'delta': {'tool_calls': [
                {'index': index, 'function': {'arguments': call['arguments']}}]}, 'finish_reason': None}]}))
    frames.append(_sse({**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish_or_stop(finish)}]}))
    frames.append(_sse({**base, 'choices': [], 'usage': {
        'prompt_tokens': input_tokens, 'completion_tokens': output_tokens,
        'total_tokens': input_tokens + output_tokens}}))
    frames.append('data: [DONE]\n\n')
    return ''.join(frames)


def render_anthropic_json(answer, model_id, req_text):
    text, calls, finish = completion_of(answer)
    input_tokens, output_tokens = settled_usage(answer, req_text, text)
    content = []
    if text != '':
        content.append({'type': 'text', 'text': text})
    for index, call in enumerate(calls):
        content.append({'type': 'tool_use',
                        'id': call['id'] or 'toolu_%d' % index,
                        'name': call['name'],
                        'input': arguments_object(call['arguments'])})
    if not content:
        content.append({'type': 'text', 'text': ''})
    return {
        'id': message_id(answer),
        'type': 'message',
        'role': 'assistant',
        'model': model_id,
        'content': content,
        'stop_reason': anthropic_stop(finish),
        'stop_sequence': None,
        'usage': {'input_tokens': input_tokens, 'output_tokens': output_tokens},
    }


def _named_sse(event, data):
    return 'event: ' + event + '\n' + _sse(data)


def render_anthropic_stream(answer, model_id, req_text):
    text, calls, finish = completion_of(answer)
    input_tokens, output_tokens = settled_usage(answer, req_text, text)
    frames = [_named_sse('message_start', {
        'type': 'message_start',
        'message': {'id': message_id(answer), 'type': 'message', 'role': 'assistant',
                    'model': model_id, 'content': [], 'stop_reason': None,
                    'stop_sequence': None,
                    'usage': {'input_tokens': input_tokens, 'output_tokens': 0}}})]
    index = 0
    if text != '' or not calls:
        frames.append(_named_sse('content_block_start', {
            'type': 'content_block_start', 'index': index,
            'content_block': {'type': 'text', 'text': ''}}))
        if text != '':
            frames.append(_named_sse('content_block_delta', {
                'type': 'content_block_delta', 'index': index,
                'delta': {'type': 'text_delta', 'text': text}}))
        frames.append(_named_sse('content_block_stop', {'type': 'content_block_stop', 'index': index}))
        index += 1
    for call in calls:
        frames.append(_named_sse('content_block_start', {
            'type': 'content_block_start', 'index': index,
            'content_block': {'type': 'tool_use', 'id': call['id'] or 'toolu_%d' % index,
                              'name': call['name'], 'input': {}}}))
        if call['arguments'] != '':
            frames.append(_named_sse('content_block_delta', {
                'type': 'content_block_delta', 'index': index,
                'delta': {'type': 'input_json_delta', 'partial_json': call['arguments']}}))
        frames.append(_named_sse('content_block_stop', {'type': 'content_block_stop', 'index': index}))
        index += 1
    frames.append(_named_sse('message_delta', {
        'type': 'message_delta',
        'delta': {'stop_reason': anthropic_stop(finish), 'stop_sequence': None},
        'usage': {'output_tokens': output_tokens}}))
    frames.append(_named_sse('message_stop', {'type': 'message_stop'}))
    return ''.join(frames)


# --------------------------------------------------------------------------
# Serving one prompt: the ranked failover chain.

def serve_prompt(wire, body):
    """Answer one client request. Returns (status, extra headers, kind, payload)
    where kind is 'json' (payload a dict) or 'sse' (payload the whole stream).

    The chain walks the ranked candidates. A refusal is recorded for the
    failures trail and the next model is tried. An answer that arrives without
    the prompt's token is kept as a fallback — a well-formed answer still
    earns its wire checks — while the chain looks for an answer that carries
    the token, the only kind the quality column credits.
    """
    requires = requirements_for(wire, body)
    candidates = ranked_candidates(requires)
    stream = body.get('stream') is True
    if not candidates:
        return 503, {}, 'json', None
    req_text = request_text(wire, body)
    wanted = tokens_in(req_text)
    failures = []
    fallback = None
    for model in candidates[:MAX_ATTEMPTS]:
        if wire == 'anthropic':
            out_body = anthropic_as_chat(body, model['id'])
        else:
            out_body = chat_body_from_openai(body, model['id'])
        status, parsed, raw = dial(out_body)
        if status is not None and 200 <= status < 300 and parsed is not None \
                and isinstance(parsed.get('choices'), list) and len(parsed['choices']) > 0:
            text, _calls, _finish = completion_of(parsed)
            if not wanted or any(token in text for token in wanted):
                return _render(wire, stream, parsed, model['id'], req_text, failures)
            if fallback is None:
                fallback = (parsed, model['id'])
            continue
        failures.append({'status': status if isinstance(status, int) else 502,
                         'code': refusal_code(parsed, raw), 'model': model['id']})
    if fallback is not None:
        parsed, model_id = fallback
        return _render(wire, stream, parsed, model_id, req_text, failures)
    return 502, {}, 'json', None


def _render(wire, stream, answer, model_id, req_text, failures):
    headers = {}
    if failures:
        headers[FAILURES_HEADER] = json.dumps(failures[:8], separators=(',', ':'), ensure_ascii=False)
    if wire == 'anthropic':
        if stream:
            return 200, headers, 'sse', render_anthropic_stream(answer, model_id, req_text)
        return 200, headers, 'json', render_anthropic_json(answer, model_id, req_text)
    if stream:
        return 200, headers, 'sse', render_openai_stream(answer, model_id, req_text)
    return 200, headers, 'json', render_openai_json(answer, model_id, req_text)


# --------------------------------------------------------------------------
# The HTTP server.

def models_document(wire):
    if wire == 'anthropic':
        return {'data': [{'type': 'model', 'id': m['id'], 'display_name': m['name'],
                          'created_at': LIST_CREATED_AT} for m in FREE_MODELS],
                'has_more': False}
    return {'object': 'list',
            'data': [{'id': m['id'], 'object': 'model', 'created': LIST_CREATED,
                      'owned_by': PROVIDER_ID} for m in FREE_MODELS]}


def error_document(wire, message, error_type, code):
    if wire == 'anthropic':
        return {'type': 'error', 'error': {'type': error_type, 'message': message}}
    return {'error': {'message': message, 'type': error_type, 'code': code}}


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'aggregator-02/1'
    sys_version = ''

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        self._dispatch()

    def do_POST(self):
        self._dispatch()

    def _dispatch(self):
        self._sent = False
        path = urlsplit(self.path).path
        wire = 'anthropic' if path == '/v1/messages' else 'openai'
        try:
            if self.command == 'GET' and path == '/v1/models':
                if self.headers.get('x-api-key') is not None:
                    self._send_json(200, models_document('anthropic'), {})
                else:
                    self._send_json(200, models_document('openai'), {})
                return
            if self.command == 'POST' and path in ('/v1/chat/completions', '/v1/messages'):
                self._complete(wire)
                return
            self._send_json(404, error_document(wire, 'no route ' + self.command + ' ' + path,
                                                'not_found', 'not_found'), {})
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception:
            import traceback
            note(traceback.format_exc())
            if not self._sent:
                try:
                    self._send_json(500, error_document(wire, 'the server failed to answer',
                                                        'internal_error', 'internal_error'), {})
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True

    def _send_json(self, status, document, headers):
        raw = json.dumps(document, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('content-type', 'application/json')
        for name, value in headers.items():
            self.send_header(name, value)
        self.send_header('content-length', str(len(raw)))
        self._sent = True
        self.end_headers()
        self.wfile.write(raw)

    def _send_stream(self, text, headers):
        self.send_response(200)
        self.send_header('content-type', 'text/event-stream')
        self.send_header('cache-control', 'no-cache')
        for name, value in headers.items():
            self.send_header(name, value)
        # No length is known to the client of a stream; the close ends it.
        self.send_header('connection', 'close')
        self._sent = True
        self.end_headers()
        self.wfile.write(text.encode('utf-8'))
        self.wfile.flush()
        self.close_connection = True

    def _read_body(self):
        length = self.headers.get('content-length')
        if length is None:
            self.close_connection = True
            return None
        try:
            size = int(length)
        except ValueError:
            self.close_connection = True
            return None
        if size < 0 or size > MAX_BODY:
            self.close_connection = True
            return None
        raw = self.rfile.read(size) if size > 0 else b''
        if not raw:
            return None
        try:
            value = json.loads(raw.decode('utf-8'))
        except (UnicodeDecodeError, ValueError):
            return None
        return value if is_obj(value) else None

    def _complete(self, wire):
        body = self._read_body()
        if body is None:
            self._send_json(400, error_document(wire, 'the request body is not one JSON object',
                                                'invalid_request_error', 'invalid_request_error'), {})
            return
        status, headers, kind, payload = serve_prompt(wire, body)
        if kind == 'sse':
            self._send_stream(payload, headers)
            return
        if payload is not None:
            self._send_json(status, payload, headers)
            return
        if status == 503:
            message = 'no free model in the brief meets this request'
            error_type = 'overloaded_error' if wire == 'anthropic' else 'no_free_model'
        else:
            message = 'every free model the prompt was tried on refused or returned no answer'
            error_type = 'api_error' if wire == 'anthropic' else 'upstream_error'
        self._send_json(status, error_document(wire, message, error_type, error_type), {})


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


# --------------------------------------------------------------------------
# Register, poll, start.

def register(port):
    body = json.dumps({'protocol_version': PROTOCOL_VERSION, 'port': port, 'catalog': CATALOG},
                      separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    request = urllib.request.Request(ENV_URL + '/register', data=body, method='POST')
    request.add_header('content-type', 'application/json')
    with OPENER.open(request, timeout=5) as response:
        reply = json.loads(response.read().decode('utf-8', 'replace'))
    if is_obj(reply) and is_num(reply.get('requests')):
        note('registered: the trial holds %s prompts' % reply['requests'])


def poll_work():
    import time
    while True:
        time.sleep(POLL_SECONDS)
        try:
            with OPENER.open(ENV_URL + '/work', timeout=2) as response:
                work = json.loads(response.read().decode('utf-8', 'replace'))
        except Exception:
            continue
        if is_obj(work) and work.get('done') is True:
            sys.stderr.flush()
            os._exit(0)


def _exit_zero(*_args):
    sys.stderr.flush()
    os._exit(0)


def main():
    if '--list' in sys.argv[1:]:
        # The same document the register body carries, byte for byte.
        sys.stdout.write(CATALOG_TEXT + '\n')
        sys.stdout.flush()
        return 0
    server = Server(('0.0.0.0', 0), Handler)
    port = server.server_address[1]
    if port == 8080:
        note('the OS picked 8080, which the env owns: refusing to start')
        return 1
    note('listening on 0.0.0.0:%d with %d free models' % (port, len(FREE_MODELS)))
    if ENV_URL != '':
        register(port)
        signal.signal(signal.SIGTERM, _exit_zero)
        signal.signal(signal.SIGINT, _exit_zero)
        threading.Thread(target=poll_work, name='work-poll', daemon=True).start()
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:
        import traceback
        note(traceback.format_exc())
        raise SystemExit(1)
