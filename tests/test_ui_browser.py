"""Real headless Chromium frontend tests with explicit mocked API responses.

These test DOM events, themes, layout and workspace portals. They are NOT live
server/model tests. Actual FastAPI handlers are tested in test_unified_aexn.py.
"""
import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BROWSER = Path('C:/Program Files/Google/Chrome/Application/chrome.exe')
if not BROWSER.exists():
    BROWSER = Path('C:/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe')

BOOTSTRAP = r'''
<script>
window.frontendErrors = [];
window.addEventListener('error', e => frontendErrors.push(e.message));
window.addEventListener('unhandledrejection', e => frontendErrors.push(String(e.reason)));
window.testRequests = [];
window.EventSource = class { constructor() {} close() {} };
window.confirm = () => true;
window.fetch = async (url, options = {}) => {
  let body = options.body ? JSON.parse(options.body) : {};
  testRequests.push({ url: String(url), body });
  let data = {};
  if (url === '/api/autonomy/status') data = {mode:'OFF',status:'stopped',goals:[],resources:{available:true},remaining_seconds:0};
  else if (url === '/api/brain') data = {trained:false,steps:0};
  else if (url === '/api/sessions') data = {sessions:[]};
  else if (url === '/api/sessions/new') data = {session_id:'frontend-test'};
  else if (String(url).includes('/messages')) data = {status:'ok'};
  else if (url === '/api/v2/chat') {
    const answer = 'Hello from AEXN AI';
    const events = [ {type:'stage',data:'Understanding your request'},
      {type:'token',data:'Hello from '}, {type:'token',data:'AEXN AI'}, {type:'done',data:answer} ];
    return new Response(events.map(e => 'data: '+JSON.stringify(e)+'\n\n'.replaceAll('\\n','\n')).join(''),
      {headers:{'Content-Type':'text/event-stream'}});
  } else if (url === '/api/coding/workspace') data = {workspace:{root_path:'Test project',workspace_name:'Test project',permission_level:'EDIT'}};
  else if (String(url).includes('/api/coding/tree')) data = {tree:{name:'Test project',children:[{name:'main.py',path:'main.py',type:'file',extension:'.py'}]}};
  else if (String(url).startsWith('/api/coding/file?')) data = {file:{path:'main.py',filename:'main.py',syntax:'python',lines_count:2,content:'value = 1\n'.replace('\\n','\n')}};
  else if (url === '/api/coding/file/save') data = {status:'ok',result:{success:true}};
  else if (url === '/api/coding/git/diff') data = {diff:'--- a/main.py\n+++ b/main.py\n+value = 2'};
  return new Response(JSON.stringify(data), {headers:{'Content-Type':'application/json'}});
};
</script>
'''

RUNNER = r'''
<pre id="frontend-test-results" style="position:fixed;right:0;top:0;z-index:99999;background:white;color:black;font-size:10px">pending</pre>
<script>
window.addEventListener('load', () => setTimeout(async () => {
  const results = [];
  const check = (name, condition) => {results.push({name,pass:!!condition}); if (!condition) throw new Error(name);};
  const delay = () => new Promise(resolve => setTimeout(resolve,30));
  try {
    const composer = document.getElementById('composer-input');
    check('one assistant composer', document.querySelectorAll('#composer-input').length === 1 && !document.getElementById('select-active-model'));
    const theme = document.documentElement.dataset.theme;
    document.getElementById('btn-theme').click();
    check('theme toggle', document.documentElement.dataset.theme !== theme);
    composer.value = 'hi';
    document.getElementById('btn-send').click();
    await delay(); await delay();
    check('Send reaches chat API', testRequests.some(r => r.url === '/api/v2/chat' && r.body.messages.at(-1).content === 'hi'));
    check('stream renders response', document.querySelector('.assistant-body').textContent.includes('Hello from AEXN AI'));
    composer.value = 'hello again';
    composer.dispatchEvent(new KeyboardEvent('keydown', {key:'Enter',bubbles:true,cancelable:true}));
    await delay(); await delay();
    check('Enter sends', testRequests.filter(r => r.url === '/api/v2/chat').length === 2);
    check('coding workspace removed', !document.getElementById('view-coding-workspace') && !document.getElementById('tab-coding'));
    check('chat sends no coding workspace requests', !testRequests.some(r => String(r.url).startsWith('/api/coding/')));
    check('no JavaScript errors', frontendErrors.length === 0);
    check('viewport has no horizontal overflow', document.documentElement.scrollWidth <= window.innerWidth);
    check('mobile navigation starts closed', window.innerWidth > 768 || document.getElementById('sidebar').classList.contains('collapsed'));
  } catch (error) {results.push({name:'exception',pass:false,error:String(error)});}
  document.getElementById('frontend-test-results').textContent=JSON.stringify({results,errors:frontendErrors});
},100));
</script>
'''


@pytest.mark.parametrize('width,height', [(1440, 1000), (390, 844)])
def test_actual_frontend_in_headless_browser(tmp_path, width, height):
    if not BROWSER.exists():
        pytest.skip('Installed Brave/Chromium browser required')
    html = (ROOT / 'app/ui/templates/index.html').read_text(encoding='utf-8')
    html = html.replace('"/static/', '"' + (ROOT / 'app/ui/static').as_uri() + '/')
    html = html.replace('<script src=', BOOTSTRAP + '<script src=', 1)
    html = html.replace('</body>', RUNNER + '</body>')
    fixture = tmp_path / 'frontend.html'
    fixture.write_text(html, encoding='utf-8')
    result = subprocess.run([str(BROWSER), '--headless', '--disable-gpu', '--no-first-run',
        '--force-device-scale-factor=1',
        '--no-default-browser-check', '--disable-background-networking',
        '--disable-crash-reporter', '--disable-breakpad',
        '--user-data-dir=' + str(tmp_path / 'browser-profile'),
        '--window-size=' + str(width) + ',' + str(height), '--virtual-time-budget=2500',
        '--screenshot=' + str(tmp_path / 'frontend.png'), '--dump-dom', fixture.as_uri()],
        capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=45)
    match = re.search(r'<pre id="frontend-test-results"[^>]*>(.*?)</pre>', result.stdout, re.S)
    assert match and match.group(1) != 'pending', result.stderr[-2000:]
    data = json.loads(match.group(1))
    assert all(item['pass'] for item in data['results']), data
