#!/usr/bin/env python3
"""
Total Recall — ChatGPT Conversation Scraper

Uses Playwright to get an authenticated browser session, then fetches
conversations via ChatGPT's backend API (no DOM scraping needed).

Usage:
  # List all conversations (titles + IDs)
  python3 scrape_chatgpt.py --list

  # List conversations in a specific ChatGPT project
  python3 scrape_chatgpt.py --list --chatgpt-project "My Project"

  # Scrape a single conversation by URL or ID
  python3 scrape_chatgpt.py --url "https://chatgpt.com/c/abc123" --project myproject
  python3 scrape_chatgpt.py --url abc123 --project myproject

  # Scrape all conversations in a ChatGPT project
  python3 scrape_chatgpt.py --chatgpt-project "My Project" --project myproject

  # Dry run (extract and save JSON but don't import)
  python3 scrape_chatgpt.py --url "https://chatgpt.com/c/abc123" --project myproject --dry-run

  # Stage for review (save to staging folder, don't import)
  python3 scrape_chatgpt.py --chatgpt-project "My Project" --project myproject --stage

  # Scrape without downloading images/assets
  python3 scrape_chatgpt.py --chatgpt-project "My Project" --project myproject --stage --no-assets
"""

import json
import os
import sys
import time
import re
import argparse

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'parsed_chats', 'chatgpt')
STAGING_DIR = os.path.join(os.path.dirname(__file__), 'staging', 'chatgpt')


def extract_conversation_id(url_or_id):
    """Extract conversation ID from a URL or bare ID."""
    if '/' in url_or_id:
        # URL like https://chatgpt.com/c/abc123 or /c/abc123
        match = re.search(r'/c/([a-f0-9-]+)', url_or_id)
        if match:
            return match.group(1)
        # GPT URLs like /g/abc123
        match = re.search(r'/g/([a-f0-9-]+)', url_or_id)
        if match:
            return match.group(1)
    return url_or_id


def get_browser(headless=False):
    """
    Launch Playwright with a persistent profile for ChatGPT.
    First run: opens a browser window for you to log in manually.
    Subsequent runs: reuses the saved session.
    """
    from playwright.sync_api import sync_playwright

    profile_dir = os.path.expanduser("~/.claude/playwright-chatgpt")
    os.makedirs(profile_dir, exist_ok=True)

    pw = sync_playwright().start()
    browser = pw.chromium.launch_persistent_context(
        user_data_dir=profile_dir,
        headless=headless,
        channel="msedge",
        args=["--disable-blink-features=AutomationControlled"],
    )

    # Navigate to ChatGPT so cookies are available for API calls
    page = browser.new_page()
    page.goto("https://chatgpt.com/", wait_until="domcontentloaded", timeout=30000)
    time.sleep(5)

    try:
        page.wait_for_load_state("networkidle", timeout=10000)
    except Exception:
        pass
    time.sleep(2)

    # Check for login page
    try:
        body_text = page.evaluate('() => document.body.innerText.substring(0, 500)')
    except Exception:
        time.sleep(3)
        body_text = page.evaluate('() => document.body.innerText.substring(0, 500)')

    if "Log in" in body_text and "Sign up" in body_text:
        print("\nNot logged in to ChatGPT.")
        print("  A browser window should be open -- please log in manually.")
        print("  Waiting for login (checking every 5 seconds)...")

        for attempt in range(120):
            time.sleep(5)
            try:
                body_text = page.evaluate('() => document.body.innerText.substring(0, 500)')
                if "Log in" not in body_text or "New chat" in body_text:
                    has_convos = page.evaluate(
                        '() => document.querySelectorAll("a[href*=\'/c/\']").length > 0'
                    )
                    if has_convos:
                        print("  Login detected! Continuing...")
                        break
            except Exception:
                pass
            if attempt % 6 == 0 and attempt > 0:
                print(f"  Still waiting... ({attempt * 5}s)")

        page.goto("https://chatgpt.com/", wait_until="networkidle", timeout=30000)
        time.sleep(3)

    # Capture auth headers from real network requests
    auth_headers = capture_auth_headers(page)

    # Stash on page object so api_fetch can use them automatically
    page._chatgpt_auth_headers = auth_headers

    # Verify we can hit the backend API
    test = api_fetch(page, '/me')
    if isinstance(test, dict) and not test.get('error'):
        name = (test.get('name') or test.get('email')
                or test.get('username') or test.get('id', 'unknown'))
        print(f'  Authenticated as: {name}')

    # Keep page open for API calls
    return pw, browser, page


def capture_auth_headers(page):
    """Intercept a real ChatGPT API request to capture the auth headers."""
    captured = {}

    def on_request(request):
        if 'backend-api' in request.url and not captured:
            for k, v in request.headers.items():
                if k.lower() in ('authorization', 'oai-device-id', 'oai-language',
                                 'oai-client-build-number', 'oai-client-version'):
                    captured[k] = v

    page.on('request', on_request)

    # Trigger a real API call by navigating to ChatGPT (sidebar loads conversations)
    page.goto("https://chatgpt.com/", wait_until="networkidle", timeout=30000)
    time.sleep(3)

    page.remove_listener('request', on_request)

    if captured:
        print(f'  Captured {len(captured)} auth headers')
    else:
        print('  WARNING: could not capture auth headers from network requests')

    return captured


def api_fetch(page, endpoint):
    """Fetch from ChatGPT backend API using captured auth headers."""
    headers = getattr(page, '_chatgpt_auth_headers', {})
    result = page.evaluate("""
    async ([endpoint, headers]) => {
        const resp = await fetch('https://chatgpt.com/backend-api' + endpoint, {
            credentials: 'include',
            headers: headers
        });
        if (!resp.ok) return {error: true, status: resp.status, text: await resp.text()};
        return await resp.json();
    }
    """, [endpoint, headers])
    if isinstance(result, dict) and result.get('error'):
        raise RuntimeError(f"API {endpoint} returned {result['status']}: {result.get('text', '')[:200]}")
    return result


def list_projects(page):
    """List ChatGPT projects via the sidebar API."""
    try:
        data = api_fetch(page, '/gizmos/snorlax/sidebar?owned_only=true&conversations_per_gizmo=5&limit=50')
        projects = []
        items = data.get('items', [])
        for item in items:
            # Structure: item.gizmo.gizmo.id, item.gizmo.gizmo.short_url
            inner = item.get('gizmo', {}).get('gizmo', {})
            gid = inner.get('id', '')
            # Name from display.name, falling back to short_url
            name = inner.get('display', {}).get('name', '')
            if not name:
                # short_url is like "g-p-{hash}-project-name", extract the human part
                short = inner.get('short_url', '')
                # Strip the g-p-{hash}- prefix
                parts = short.split('-', 6)  # g-p-hash-name-parts
                if len(parts) > 3:
                    name = ' '.join(parts[3:]).replace('-', ' ').title()
                else:
                    name = short or 'Untitled'
            if gid.startswith('g-p-'):
                projects.append({
                    'id': gid,
                    'name': name,
                    'conversations': item.get('conversations', []),
                })
        return projects
    except RuntimeError as e:
        print(f'  Sidebar API error: {e}')
        return []


def list_conversations_in_project(page, gizmo_id, limit=100):
    """List conversations within a specific ChatGPT project (gizmo)."""
    conversations = []
    cursor = 0

    while len(conversations) < limit:
        data = api_fetch(page, f'/gizmos/{gizmo_id}/conversations?cursor={cursor}')
        items = data.get('items', data.get('conversations', []))
        if not items:
            break

        for item in items:
            conv = {
                'id': item.get('id', item.get('conversation_id', '')),
                'title': item.get('title', 'Untitled'),
                'url': f"https://chatgpt.com/c/{item.get('id', item.get('conversation_id', ''))}",
                'create_time': item.get('create_time'),
                'update_time': item.get('update_time'),
                'gizmo_id': gizmo_id,
            }
            conversations.append(conv)

        # Check for more pages
        if len(items) < 28:
            break
        cursor += len(items)
        time.sleep(0.5)

    return conversations[:limit]


def list_conversations(page, chatgpt_project=None, limit=100):
    """List conversations. If chatgpt_project specified, filter to that project.
    Otherwise list from all projects."""
    projects = list_projects(page)

    if chatgpt_project:
        # Find matching project
        matched = None
        for p in projects:
            if chatgpt_project.lower() in p['name'].lower():
                matched = p
                break
        if not matched:
            print(f'  WARNING: ChatGPT project "{chatgpt_project}" not found.')
            print(f'  Available projects: {", ".join(p["name"] for p in projects)}')
            return []
        projects = [matched]

    conversations = []
    for proj in projects:
        convos = list_conversations_in_project(page, proj['id'], limit=limit - len(conversations))
        for c in convos:
            c['project_name'] = proj['name']
        conversations.extend(convos)
        if len(conversations) >= limit:
            break

    # Also get top-level (non-project) conversations
    if not chatgpt_project:
        try:
            data = api_fetch(page, f'/conversations?offset=0&limit={min(limit, 28)}&order=updated')
            for item in data.get('items', []):
                conversations.append({
                    'id': item['id'],
                    'title': item.get('title', 'Untitled'),
                    'url': f"https://chatgpt.com/c/{item['id']}",
                    'create_time': item.get('create_time'),
                    'update_time': item.get('update_time'),
                    'project_name': '(none)',
                })
        except RuntimeError:
            pass

    return conversations[:limit]


def resolve_asset_url(page, asset_id, conv_id):
    """Resolve a file ID to a downloadable URL by extracting it from the rendered page."""
    try:
        url = page.evaluate(f"""
        () => {{
            const imgs = document.querySelectorAll('img');
            for (const img of imgs) {{
                if (img.src && img.src.includes('{asset_id}')) return img.src;
            }}
            return null;
        }}
        """)
        if url:
            return url
    except Exception:
        pass
    # Fallback: construct estuary URL (may need sig from page)
    return f'https://chatgpt.com/backend-api/estuary/content?id={asset_id}&p=fs&cid=1'


def download_asset(page, asset_url, save_dir, filename):
    """Download an asset (image, file) via the authenticated session."""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    if os.path.exists(filepath):
        return filepath  # already downloaded

    try:
        headers = getattr(page, '_chatgpt_auth_headers', {})
        result = page.evaluate("""
        async ([url, headers]) => {
            try {
                const resp = await fetch(url, { credentials: 'include', headers: headers });
                if (!resp.ok) return {error: true, status: resp.status};
                const blob = await resp.blob();
                const reader = new FileReader();
                return new Promise((resolve) => {
                    reader.onloadend = () => resolve({data: reader.result, type: blob.type, size: blob.size});
                    reader.readAsDataURL(blob);
                });
            } catch(e) { return {error: true, message: e.message}; }
        }
        """, [asset_url, headers])

        if isinstance(result, dict) and result.get('error'):
            print(f'    Asset download failed: {result}')
            return None

        # Decode base64 data URL
        import base64
        data_url = result.get('data', '')
        if ',' in data_url:
            b64_data = data_url.split(',', 1)[1]
            with open(filepath, 'wb') as f:
                f.write(base64.b64decode(b64_data))
            print(f'    Asset saved: {filename} ({result.get("size", 0)} bytes)')
            return filepath
    except Exception as e:
        print(f'    Asset download error: {e}')
    return None


def scrape_conversation(page, conversation_id, download_assets=True):
    """Fetch full conversation via backend API. Returns (title, messages, metadata)."""
    conv_id = extract_conversation_id(conversation_id)
    data = api_fetch(page, f'/conversation/{conv_id}')

    title = data.get('title', 'Untitled')
    create_time = data.get('create_time')

    # Prepare assets directory
    safe_title = "".join(c if c.isalnum() or c in '-_ ' else '' for c in title)
    safe_title = safe_title.strip().replace(' ', '-')[:80]
    assets_dir = os.path.join(STAGING_DIR, '_assets', safe_title) if download_assets else None

    # Navigate to the conversation page so images render in DOM (needed for URL resolution)
    if download_assets:
        try:
            page.goto(f'https://chatgpt.com/c/{conv_id}', wait_until='domcontentloaded', timeout=30000)
            time.sleep(5)
        except Exception:
            pass

    # ChatGPT stores messages in a tree structure via `mapping`
    mapping = data.get('mapping', {})

    # Build ordered message list by walking the tree
    messages = []
    asset_index = 0
    for node_id, node in mapping.items():
        msg = node.get('message')
        if not msg:
            continue

        author_role = msg.get('author', {}).get('role', '')
        content = msg.get('content', {})
        content_type = content.get('content_type', '')
        meta = msg.get('metadata', {})

        # Keep user, assistant, and tool messages that carry useful content
        if author_role == 'system':
            continue
        tool_content_types = ('multimodal_text', 'image_asset_pointer', 'execution_output',
                              'tether_browsing_display', 'tether_quote')
        if author_role == 'tool' and content_type not in tool_content_types:
            continue

        # Tool messages attribute to assistant in output
        display_role = 'assistant' if author_role == 'tool' else author_role

        # Handle user file attachments (PDFs, docs, code files)
        parts = content.get('parts', [])
        text_parts = []
        msg_assets = []
        attachments = meta.get('attachments', [])
        if attachments and download_assets:
            for att in attachments:
                att_id = att.get('id', '')
                att_name = att.get('name', 'unknown')
                att_mime = att.get('mime_type', '')
                att_size = att.get('size', 0)
                # Try to download the attachment
                att_url = None
                if att_id:
                    try:
                        dl_info = api_fetch(page, f'/files/{att_id}/download')
                        att_url = dl_info.get('download_url') or dl_info.get('url')
                    except Exception:
                        # Try estuary
                        att_url = resolve_asset_url(page, att_id, conv_id)
                if att_url and assets_dir:
                    local_path = download_asset(page, att_url, assets_dir, att_name)
                    if local_path:
                        rel_path = os.path.relpath(local_path, STAGING_DIR)
                        text_parts.append(f'[attachment: {rel_path}]')
                        msg_assets.append({
                            'type': 'attachment',
                            'asset_id': att_id,
                            'local_path': rel_path,
                            'filename': att_name,
                            'mime_type': att_mime,
                            'size': att_size,
                        })
                        asset_index += 1
                        continue
                text_parts.append(f'[attachment: {att_name} ({att_mime}, {att_size} bytes)]')

        # Handle code interpreter execution output
        if content_type == 'execution_output':
            agg = meta.get('aggregate_result', {})
            code = agg.get('code', '')
            final_output = agg.get('final_expression_output', '')
            agg_messages = agg.get('messages', [])
            if code:
                text_parts.append(f'```python\n{code}\n```')
            if final_output:
                text_parts.append(f'Output: {final_output}')
            # Check for generated files in messages
            for am in agg_messages:
                if isinstance(am, dict) and am.get('message_type') == 'image':
                    img_url = am.get('image_url', '')
                    if img_url and download_assets and assets_dir:
                        fname = f'{asset_index:03d}_interpreter_output.png'
                        local_path = download_asset(page, img_url, assets_dir, fname)
                        if local_path:
                            rel_path = os.path.relpath(local_path, STAGING_DIR)
                            text_parts.append(f'[interpreter output: {rel_path}]')
                            msg_assets.append({'type': 'interpreter_output', 'local_path': rel_path})
                            asset_index += 1

        # Handle browsing content
        if content_type == 'tether_browsing_display':
            text_parts.append('[web browsing result]')
        if content_type == 'tether_quote':
            quote_text = '\n'.join(p if isinstance(p, str) else p.get('text', '') for p in parts)
            if quote_text.strip():
                text_parts.append(f'[web quote] {quote_text.strip()}')
        for part in parts:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict):
                # Could be an image, code, etc.
                if part.get('content_type') == 'text':
                    text_parts.append(part.get('text', ''))
                elif part.get('content_type') == 'image_asset_pointer':
                    raw_pointer = part.get('asset_pointer', '')
                    # Extract bare file ID from various protocols
                    asset_id = raw_pointer.replace('file-service://', '').replace('sediment://', '')
                    asset_url = None
                    if asset_id and download_assets:
                        asset_url = resolve_asset_url(page, asset_id, conv_id)
                    if asset_url and assets_dir:
                        ext = '.png'  # default
                        dalle_meta = (part.get('metadata') or {}).get('dalle') or {}
                        if dalle_meta.get('prompt'):
                            ext = '.webp'
                        filename = f'{asset_index:03d}_{asset_id[:12]}{ext}'
                        local_path = download_asset(page, asset_url, assets_dir, filename)
                        if local_path:
                            rel_path = os.path.relpath(local_path, STAGING_DIR)
                            dalle_prompt = dalle_meta.get('prompt')
                            if dalle_prompt:
                                text_parts.append(f'[image: {rel_path}]\nDALL-E prompt: {dalle_prompt}')
                            else:
                                text_parts.append(f'[image: {rel_path}]')
                            msg_assets.append({
                                'type': 'image',
                                'asset_id': asset_id,
                                'local_path': rel_path,
                                'dalle_prompt': dalle_prompt,
                            })
                            asset_index += 1
                            continue
                    text_parts.append('[image]')
                elif part.get('content_type') == 'multimodal_text':
                    # User uploads or DALL-E outputs nested in multimodal_text
                    inner_parts = part.get('parts', [])
                    for ip in inner_parts:
                        if isinstance(ip, str):
                            text_parts.append(ip)
                        elif isinstance(ip, dict) and ip.get('content_type') == 'image_asset_pointer':
                            # Same download logic as top-level image_asset_pointer
                            ip_raw = ip.get('asset_pointer', '')
                            ip_asset_id = ip_raw.replace('file-service://', '').replace('sediment://', '')
                            ip_url = None
                            if ip_asset_id and download_assets:
                                ip_url = resolve_asset_url(page, ip_asset_id, conv_id)
                            if ip_url and assets_dir:
                                ip_dalle_meta = (ip.get('metadata') or {}).get('dalle') or {}
                                ip_ext = '.webp' if ip_dalle_meta.get('prompt') else '.png'
                                ip_filename = f'{asset_index:03d}_{ip_asset_id[:12]}{ip_ext}'
                                local_path = download_asset(page, ip_url, assets_dir, ip_filename)
                                if local_path:
                                    rel_path = os.path.relpath(local_path, STAGING_DIR)
                                    ip_dalle_prompt = ip_dalle_meta.get('prompt')
                                    if ip_dalle_prompt:
                                        text_parts.append(f'[image: {rel_path}]\nDALL-E prompt: {ip_dalle_prompt}')
                                    else:
                                        text_parts.append(f'[image: {rel_path}]')
                                    msg_assets.append({
                                        'type': 'image',
                                        'asset_id': ip_asset_id,
                                        'local_path': rel_path,
                                        'dalle_prompt': ip.get('metadata', {}).get('dalle', {}).get('prompt'),
                                    })
                                    asset_index += 1
                                    continue
                            text_parts.append('[image]')
                        elif isinstance(ip, dict) and ip.get('text'):
                            text_parts.append(ip['text'])
                elif 'text' in part:
                    text_parts.append(part['text'])

        text = '\n'.join(text_parts).strip()
        if not text and not msg_assets:
            continue

        # Get timestamp
        ts = msg.get('create_time')
        if ts:
            from datetime import datetime, timezone
            ts_str = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        else:
            ts_str = ''

        # Check for tool use (code interpreter, browsing, etc.)
        has_tool = author_role == 'assistant' and msg.get('metadata', {}).get('invoked_plugin') is not None
        if not has_tool:
            has_tool = any(
                n.get('message', {}).get('author', {}).get('role') == 'tool'
                for n in mapping.values()
                if n.get('parent') == node_id
            )

        entry = {
            'role': display_role,
            'text': text,
            'timestamp': ts_str,
            'has_tool_use': has_tool,
            'node_id': node_id,
            'create_time': ts or 0,
        }
        if msg_assets:
            entry['assets'] = msg_assets
        messages.append(entry)

    # Sort by create_time to get chronological order
    messages.sort(key=lambda m: m['create_time'])

    # Strip internal fields
    for msg in messages:
        del msg['node_id']
        del msg['create_time']

    total_assets = sum(len(m.get('assets', [])) for m in messages)
    metadata = {
        'conversation_id': conv_id,
        'title': title,
        'create_time': create_time,
        'model_slug': data.get('default_model_slug'),
        'gizmo_id': data.get('gizmo_id'),
        'asset_count': total_assets,
    }
    if total_assets:
        print(f'    Downloaded {total_assets} assets')

    return title, messages, metadata


def save_conversation(title, messages, project, output_dir=OUTPUT_DIR, metadata=None):
    """Save extracted messages as structured JSON for import."""
    os.makedirs(output_dir, exist_ok=True)

    safe_title = "".join(c if c.isalnum() or c in '-_ ' else '' for c in title)
    safe_title = safe_title.strip().replace(' ', '-')[:80]
    filename = f"{safe_title}.json"
    filepath = os.path.join(output_dir, filename)

    # Messages already have real timestamps from the API
    output = {
        'metadata': metadata or {},
        'messages': messages,
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f'  Saved: {filepath} ({len(messages)} messages)')
    return filepath


def import_conversation(json_path, project):
    """Import the saved JSON into the session memory DB."""
    import subprocess

    # import_parsed.py expects a flat array of messages
    # Our save format wraps them in {metadata, messages}
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

    messages = data.get('messages', data) if isinstance(data, dict) else data

    # Write a temp flat file for import_parsed.py
    flat_path = json_path.replace('.json', '_flat.json')
    with open(flat_path, 'w', encoding='utf-8') as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

    result = subprocess.run(
        ['python3', 'import_parsed.py', flat_path, '--project', project],
        cwd=os.path.dirname(__file__),
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f'  Import error: {result.stderr}', file=sys.stderr)

    # Clean up temp file
    try:
        os.remove(flat_path)
    except OSError:
        pass


def main():
    parser = argparse.ArgumentParser(
        description='Scrape ChatGPT conversations via backend API'
    )
    parser.add_argument('--list', action='store_true',
                        help='List conversations (titles + IDs)')
    parser.add_argument('--list-projects', action='store_true',
                        help='List ChatGPT projects')
    parser.add_argument('--url', default=None,
                        help='Scrape a specific conversation (URL or ID)')
    parser.add_argument('--chatgpt-project', default=None,
                        help='ChatGPT project name to filter/scrape')
    parser.add_argument('--project', default=None,
                        help='Project tag for import')
    parser.add_argument('--dry-run', action='store_true',
                        help='Extract and save JSON but do not import')
    parser.add_argument('--stage', action='store_true',
                        help='Save to staging folder for review')
    parser.add_argument('--limit', type=int, default=100,
                        help='Max conversations to list (default: 100)')
    parser.add_argument('--all', action='store_true',
                        help='Scrape all conversations across all projects')
    parser.add_argument('--headless', action='store_true',
                        help='Run browser headless (no visible window)')
    parser.add_argument('--no-assets', action='store_true',
                        help='Skip downloading images and file assets')
    args = parser.parse_args()

    if not args.list and not args.list_projects and not args.url and not args.chatgpt_project and not args.all:
        parser.print_help()
        sys.exit(1)

    print('Launching browser...')
    pw, browser, page = get_browser(headless=args.headless)

    out_dir = STAGING_DIR if args.stage else OUTPUT_DIR

    try:
        if os.environ.get('DEBUG'):
            # Sniff all backend-api requests the UI makes on page load
            seen = []
            def on_req(req):
                if 'backend-api' in req.url:
                    path = req.url.split('backend-api')[1]
                    seen.append(path)
            page.on('request', on_req)
            page.goto("https://chatgpt.com/", wait_until="networkidle", timeout=30000)
            time.sleep(5)
            page.remove_listener('request', on_req)
            print(f'\n  Backend API requests on page load ({len(seen)}):')
            for s in seen:
                print(f'    {s}')

        if args.list_projects:
            projects = list_projects(page)
            print(f'\nFound {len(projects)} projects:')
            for p in projects:
                print(f'  {p["name"][:60]:60s}  {p["id"]}')

        elif args.list:
            convos = list_conversations(page, args.chatgpt_project, limit=args.limit)
            print(f'\nFound {len(convos)} conversations:')
            for c in convos:
                from datetime import datetime, timezone
                updated = ''
                if c.get('update_time'):
                    try:
                        updated = datetime.fromtimestamp(float(c['update_time']), tz=timezone.utc).strftime('%Y-%m-%d')
                    except (ValueError, TypeError, OSError):
                        updated = str(c['update_time'])[:10]
                proj = c.get('project_name', '')
                print(f'  {updated:12s} {proj[:20]:20s} {c["title"][:45]:45s}  {c["id"]}')

        elif args.url:
            if not args.project and not args.stage:
                print('ERROR: --project is required when scraping (or use --stage)', file=sys.stderr)
                sys.exit(1)

            print(f'Scraping: {args.url}')
            title, messages, metadata = scrape_conversation(page, args.url, download_assets=not args.no_assets)
            print(f'  Title: {title}')
            print(f'  Messages: {len(messages)}')

            if messages:
                json_path = save_conversation(title, messages, args.project, out_dir, metadata)
                if not args.dry_run and not args.stage:
                    import_conversation(json_path, args.project)
            else:
                print('  No messages extracted!')

        elif args.chatgpt_project or args.all:
            if not args.project and not args.stage:
                print('ERROR: --project is required when scraping (or use --stage)', file=sys.stderr)
                sys.exit(1)

            convos = list_conversations(page, args.chatgpt_project, limit=args.limit)
            label = f'"{args.chatgpt_project}"' if args.chatgpt_project else 'all projects'
            print(f'\nFound {len(convos)} conversations in {label}')

            for i, c in enumerate(convos):
                proj_name = c.get('project_name', '')
                print(f'\n[{i+1}/{len(convos)}] [{proj_name}] {c["title"][:50]}')
                title, messages, metadata = scrape_conversation(page, c['id'], download_assets=not args.no_assets)
                print(f'  Messages: {len(messages)}')

                if messages:
                    # Use per-project subdirs when staging all
                    if args.all and args.stage:
                        safe_proj = proj_name.replace(' ', '-').replace('/', '-') or 'uncategorized'
                        conv_out_dir = os.path.join(out_dir, safe_proj)
                    else:
                        conv_out_dir = out_dir
                    json_path = save_conversation(title, messages, args.project, conv_out_dir, metadata)
                    if not args.dry_run and not args.stage:
                        import_conversation(json_path, args.project)
                else:
                    print('  No messages extracted, skipping')

                time.sleep(1)  # gentle rate limit

    finally:
        page.close()
        browser.close()
        pw.stop()

    print('\nDone.')


if __name__ == '__main__':
    main()
