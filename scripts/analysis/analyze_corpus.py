import json
import math
import re
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path

RAW_MESSAGES_PATH = Path("isaac_messages_clean.jsonl")
TURNS_PATH = Path('isaac_turns.jsonl')
SESSIONS_PATH = Path('isaac_sessions.jsonl')
JSON_OUTPUT_PATH = Path('corpus_analysis.json')
MARKDOWN_OUTPUT_PATH = Path('corpus_analysis.md')

SPEAKERS = ('Aaron', 'Isaac')
TOP_WORDS = 30
TOP_PHRASES = 25
TOP_EMOJIS = 25
TOP_DISTINCTIVE_WORDS = 30
MIN_DISTINCTIVE_COUNT = 15

URL_RE = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*")

STOPWORDS = {
    'a','an','and','are','as','at','be','because','been','but','by','can','could',
    'did','do','does','for','from','had','has','have','he','her','here','him','his',
    'how','i','if','im','in','is','it','its','just','like','me','my','no','not','of',
    'oh','on','or','our','out','she','so','that','the','their','them','then','there',
    'they','this','to','too','up','was','we','were','what','when','where','who','why',
    'with','would','yeah','yes','you','your','youre','youll','youve','dont','didnt',
    'cant','wont','thats','theres'
}


def load_jsonl(path):
    records = []
    with path.open('r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f'Invalid JSON in {path} line {line_number}: {exc}') from exc
    return records


def parse_timestamp(value):
    return datetime.fromisoformat(value) if value else None


def strip_urls(text):
    return URL_RE.sub(' ', text or '')


def tokenize(text):
    return WORD_RE.findall(strip_urls(text).lower())


def count_words(text):
    return len(tokenize(text))


def is_all_caps_message(text):
    letters = [c for c in (text or '') if c.isalpha()]
    return len(letters) >= 2 and all(c == c.upper() for c in letters)


def is_emoji_codepoint(char):
    cp = ord(char)
    ranges = (
        (0x1F300,0x1F5FF),(0x1F600,0x1F64F),(0x1F680,0x1F6FF),
        (0x1F700,0x1F77F),(0x1F780,0x1F7FF),(0x1F800,0x1F8FF),
        (0x1F900,0x1F9FF),(0x1FA00,0x1FAFF),(0x2600,0x26FF),
        (0x2700,0x27BF),(0x2300,0x23FF),(0x2B00,0x2BFF)
    )
    return any(start <= cp <= end for start, end in ranges)


def extract_emojis(text):
    if not text:
        return []
    emojis = []
    current = ''
    for char in text:
        cp = ord(char)
        if is_emoji_codepoint(char):
            if current:
                emojis.append(current)
            current = char
        elif current and (cp == 0xFE0F or 0x1F3FB <= cp <= 0x1F3FF or cp == 0x200D):
            current += char
        else:
            if current:
                emojis.append(current)
                current = ''
    if current:
        emojis.append(current)
    return emojis


def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)] if len(tokens) >= n else []


def percentile(values, percent):
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    position = (percent / 100) * (len(values) - 1)
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return float(values[lo])
    frac = position - lo
    return values[lo] + frac * (values[hi] - values[lo])


def summarize(values):
    if not values:
        return {'count':0,'mean':0.0,'median':0.0,'p90':0.0,'p95':0.0,'max':0.0}
    return {
        'count': len(values),
        'mean': statistics.mean(values),
        'median': statistics.median(values),
        'p90': percentile(values, 90),
        'p95': percentile(values, 95),
        'max': max(values),
    }


def analyze_messages(messages):
    grouped = {s: [] for s in SPEAKERS}
    for m in messages:
        if m.get('sender') in grouped:
            grouped[m['sender']].append(m)

    results = {}
    for speaker, items in grouped.items():
        word_counter = Counter()
        bigrams = Counter()
        trigrams = Counter()
        emojis = Counter()
        word_counts = []
        char_counts = []
        caps = questions = exclaims = emoji_msgs = url_msgs = 0

        for m in items:
            text = m.get('text') or ''
            tokens = tokenize(text)
            word_counter.update(tokens)
            bigrams.update(ngrams(tokens, 2))
            trigrams.update(ngrams(tokens, 3))
            em = extract_emojis(text)
            emojis.update(em)
            word_counts.append(len(tokens))
            char_counts.append(len(text))
            caps += int(is_all_caps_message(text))
            questions += int('?' in text)
            exclaims += int('!' in text)
            emoji_msgs += int(bool(em))
            url_msgs += int(bool(URL_RE.search(text)))

        n = len(items)
        results[speaker] = {
            'message_count': n,
            'total_words': sum(word_counts),
            'total_characters': sum(char_counts),
            'words_per_message': summarize(word_counts),
            'characters_per_message': summarize(char_counts),
            'all_caps_message_percent': caps / n * 100 if n else 0,
            'question_message_percent': questions / n * 100 if n else 0,
            'exclamation_message_percent': exclaims / n * 100 if n else 0,
            'emoji_message_percent': emoji_msgs / n * 100 if n else 0,
            'url_message_percent': url_msgs / n * 100 if n else 0,
            'top_words': word_counter.most_common(TOP_WORDS),
            'top_bigrams': [[' '.join(p), c] for p, c in bigrams.most_common(TOP_PHRASES)],
            'top_trigrams': [[' '.join(p), c] for p, c in trigrams.most_common(TOP_PHRASES)],
            'top_emojis': emojis.most_common(TOP_EMOJIS),
            '_word_counter': word_counter,
        }
    return results


def fragmentation_buckets(counts):
    buckets = {'1':0,'2':0,'3':0,'4':0,'5+':0}
    for count in counts:
        buckets['5+' if count >= 5 else str(count)] += 1
    total = len(counts)
    return {
        'counts': buckets,
        'percentages': {k: (v / total * 100 if total else 0.0) for k, v in buckets.items()}
    }


def analyze_turns(turns):
    grouped = {s: [] for s in SPEAKERS}
    for t in turns:
        if t.get('speaker') in grouped:
            grouped[t['speaker']].append(t)

    results = {}
    for speaker, items in grouped.items():
        messages_per_turn = [t.get('message_count', 0) for t in items]
        words_per_turn = [count_words(t.get('text') or '') for t in items]
        chars_per_turn = [len(t.get('text') or '') for t in items]
        results[speaker] = {
            'turn_count': len(items),
            'messages_per_turn': summarize(messages_per_turn),
            'words_per_turn': summarize(words_per_turn),
            'characters_per_turn': summarize(chars_per_turn),
            'message_fragmentation': fragmentation_buckets(messages_per_turn),
        }
    return results


def analyze_sessions(sessions):
    initiators = Counter()
    response_times = {s: [] for s in SPEAKERS}
    turn_counts = []
    durations_min = []
    usable = 0

    for session in sessions:
        turns = session.get('turns', [])
        if not turns:
            continue
        usable += 1
        first = turns[0].get('speaker')
        if first in SPEAKERS:
            initiators[first] += 1
        turn_counts.append(len(turns))

        start = parse_timestamp(session.get('start_time'))
        end = parse_timestamp(session.get('end_time'))
        if start and end:
            durations_min.append(max((end - start).total_seconds() / 60, 0))

        for prev, cur in zip(turns, turns[1:]):
            ps = prev.get('speaker')
            cs = cur.get('speaker')
            if ps not in SPEAKERS or cs not in SPEAKERS or ps == cs:
                continue
            prev_end = parse_timestamp(prev.get('end_time'))
            cur_start = parse_timestamp(cur.get('start_time'))
            if not prev_end or not cur_start:
                continue
            seconds = (cur_start - prev_end).total_seconds()
            if seconds >= 0:
                response_times[cs].append(seconds)

    initiation = {
        s: {
            'sessions_started': initiators[s],
            'percent': initiators[s] / usable * 100 if usable else 0.0,
        }
        for s in SPEAKERS
    }

    response_summary = {
        s: {
            'seconds': summarize(response_times[s]),
            'minutes': summarize([x / 60 for x in response_times[s]]),
        }
        for s in SPEAKERS
    }

    return {
        'session_count': usable,
        'turns_per_session': summarize(turn_counts),
        'duration_minutes': summarize(durations_min),
        'conversation_initiation': initiation,
        'response_times': response_summary,
    }


def distinctive_words(message_analysis):
    ac = message_analysis['Aaron']['_word_counter']
    ic = message_analysis['Isaac']['_word_counter']
    at = message_analysis['Aaron']['total_words']
    it = message_analysis['Isaac']['total_words']
    vocab = set(ac) | set(ic)
    aaron_ranked = []
    isaac_ranked = []

    for word in vocab:
        total = ac[word] + ic[word]
        if total < MIN_DISTINCTIVE_COUNT or word in STOPWORDS or word.isdigit():
            continue
        ar = (ac[word] + 0.5) / (at + 0.5 * max(len(ac), 1))
        ir = (ic[word] + 0.5) / (it + 0.5 * max(len(ic), 1))
        score = math.log2(ir / ar)
        row = {
            'word': word,
            'score': score,
            'isaac_count': ic[word],
            'aaron_count': ac[word],
            'total_count': total,
        }
        (isaac_ranked if score > 0 else aaron_ranked).append(row)

    isaac_ranked.sort(key=lambda x: x['score'], reverse=True)
    aaron_ranked.sort(key=lambda x: x['score'])
    return {
        'Isaac': isaac_ranked[:TOP_DISTINCTIVE_WORDS],
        'Aaron': aaron_ranked[:TOP_DISTINCTIVE_WORDS],
    }


def selected_token_counts(message_analysis):
    tokens = ['bro','bruh','lmao','lmfao','lol','omg','wtf','fr','girl','dude','pls','plz','nah','yooo','yo']
    result = {}
    for speaker in SPEAKERS:
        counter = message_analysis[speaker]['_word_counter']
        total = message_analysis[speaker]['total_words']
        result[speaker] = {
            token: {
                'count': counter[token],
                'per_1000_words': counter[token] / total * 1000 if total else 0.0,
            }
            for token in tokens
        }
    return result


def clean_message_analysis(message_analysis):
    cleaned = {}
    for speaker, data in message_analysis.items():
        cleaned[speaker] = {k: v for k, v in data.items() if not k.startswith('_')}
    return cleaned


def md_table(headers, rows):
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join('---' for _ in headers) + ' |',
    ]
    lines.extend('| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows)
    return '\n'.join(lines)


def build_markdown(analysis):
    m = analysis['messages']
    t = analysis['turns']
    s = analysis['sessions']
    tok = analysis['selected_tokens']
    dist = analysis['distinctive_words']
    lines = ['# Aaron–Isaac Corpus Analysis', '']

    lines += ['## Corpus overview', '']
    rows = []
    for speaker in SPEAKERS:
        rows.append([
            speaker,
            f"{m[speaker]['message_count']:,}",
            f"{t[speaker]['turn_count']:,}",
            f"{m[speaker]['total_words']:,}",
            f"{m[speaker]['words_per_message']['mean']:.2f}",
            f"{t[speaker]['words_per_turn']['mean']:.2f}",
            f"{t[speaker]['messages_per_turn']['mean']:.2f}",
        ])
    lines += [md_table(['Speaker','Messages','Turns','Words','Avg words/msg','Avg words/turn','Avg msgs/turn'], rows), '']

    lines += ['## Message-level style', '']
    rows = []
    for speaker in SPEAKERS:
        d = m[speaker]
        rows.append([
            speaker,
            f"{d['all_caps_message_percent']:.1f}%",
            f"{d['question_message_percent']:.1f}%",
            f"{d['exclamation_message_percent']:.1f}%",
            f"{d['emoji_message_percent']:.1f}%",
            f"{d['url_message_percent']:.1f}%",
        ])
    lines += [md_table(['Speaker','ALL CAPS','Has ?','Has !','Has emoji','Has URL'], rows), '']

    lines += ['## Message fragmentation within turns', '']
    rows = []
    for speaker in SPEAKERS:
        p = t[speaker]['message_fragmentation']['percentages']
        rows.append([speaker, f"{p['1']:.1f}%", f"{p['2']:.1f}%", f"{p['3']:.1f}%", f"{p['4']:.1f}%", f"{p['5+']:.1f}%"])
    lines += [md_table(['Speaker','1 msg','2 msgs','3 msgs','4 msgs','5+ msgs'], rows), '']

    lines += ['## Sessions and response timing', '', f"Sessions analyzed: **{s['session_count']:,}**", '']
    rows = []
    for speaker in SPEAKERS:
        init = s['conversation_initiation'][speaker]
        resp = s['response_times'][speaker]['minutes']
        rows.append([
            speaker,
            f"{init['sessions_started']:,}",
            f"{init['percent']:.1f}%",
            f"{resp['median']:.2f} min",
            f"{resp['mean']:.2f} min",
            f"{resp['p90']:.2f} min",
        ])
    lines += [md_table(['Speaker','Sessions started','Initiation share','Median response','Mean response','P90 response'], rows), '']

    lines += ['## Selected slang / style tokens', '']
    rows = []
    for token in tok['Aaron']:
        rows.append([
            token,
            tok['Aaron'][token]['count'],
            f"{tok['Aaron'][token]['per_1000_words']:.2f}",
            tok['Isaac'][token]['count'],
            f"{tok['Isaac'][token]['per_1000_words']:.2f}",
        ])
    lines += [md_table(['Token','Aaron count','Aaron / 1k','Isaac count','Isaac / 1k'], rows), '']

    lines += ['## Most Isaac-distinctive words', '', 'Positive scores mean proportionally more common in Isaac\'s corpus.', '']
    rows = [[x['word'], f"{x['score']:.2f}", x['isaac_count'], x['aaron_count']] for x in dist['Isaac']]
    lines += [md_table(['Word','log2 rate ratio','Isaac count','Aaron count'], rows), '']

    lines += ['## Most Aaron-distinctive words', '']
    rows = [[x['word'], f"{x['score']:.2f}", x['aaron_count'], x['isaac_count']] for x in dist['Aaron']]
    lines += [md_table(['Word','log2 Isaac/Aaron','Aaron count','Isaac count'], rows), '']

    for speaker in SPEAKERS:
        lines += [f'## {speaker}: common language', '', '### Top words', '']
        lines += [md_table(['Word','Count'], m[speaker]['top_words']), '']
        lines += ['### Top bigrams', '', md_table(['Phrase','Count'], m[speaker]['top_bigrams']), '']
        lines += ['### Top trigrams', '', md_table(['Phrase','Count'], m[speaker]['top_trigrams']), '']
        lines += ['### Top emojis', '']
        lines += [md_table(['Emoji','Count'], m[speaker]['top_emojis']) if m[speaker]['top_emojis'] else '_No emoji detected._', '']

    return '\n'.join(lines)


def main():
    print('Loading corpus...')
    messages = load_jsonl(RAW_MESSAGES_PATH)
    turns = load_jsonl(TURNS_PATH)
    sessions = load_jsonl(SESSIONS_PATH)
    print(f'Loaded {len(messages):,} messages, {len(turns):,} turns, {len(sessions):,} sessions.')

    print('Analyzing messages...')
    ma = analyze_messages(messages)
    print('Analyzing turns...')
    ta = analyze_turns(turns)
    print('Analyzing sessions and response times...')
    sa = analyze_sessions(sessions)
    print('Calculating distinctive vocabulary...')
    dw = distinctive_words(ma)
    print('Counting selected slang/style tokens...')
    st = selected_token_counts(ma)

    analysis = {
        'source_files': {
            'messages': str(RAW_MESSAGES_PATH),
            'turns': str(TURNS_PATH),
            'sessions': str(SESSIONS_PATH),
        },
        'record_counts': {
            'messages': len(messages),
            'turns': len(turns),
            'sessions': len(sessions),
        },
        'messages': clean_message_analysis(ma),
        'turns': ta,
        'sessions': sa,
        'selected_tokens': st,
        'distinctive_words': dw,
    }

    with JSON_OUTPUT_PATH.open('w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    MARKDOWN_OUTPUT_PATH.write_text(build_markdown(analysis), encoding='utf-8')

    print('\nDone.')
    print(f'JSON analysis:   {JSON_OUTPUT_PATH}')
    print(f'Readable report: {MARKDOWN_OUTPUT_PATH}')
    print('\nOpen corpus_analysis.md first.')


if __name__ == '__main__':
    main()
