"""Real browser rendering of the dashboard with explicit mock API/EventSource.

Uses actual production JS/CSS; mocks only the backend transport. Server-side
worker evidence and full cycles are exercised separately by integration tests.
"""
import json
import re
import subprocess
from pathlib import Path

import pytest

from tests.test_ui_browser import BROWSER, ROOT


@pytest.mark.parametrize("width,height", [(1440, 1000), (390, 844)])
def test_autonomy_controls_and_safe_event_rendering(tmp_path, width, height):
    if not BROWSER.exists(): pytest.skip("Installed Chromium browser required")
    css = (ROOT / "app/ui/static/css/autonomy.css").as_uri()
    js = (ROOT / "app/ui/static/js/autonomy.js").as_uri()
    html = '''<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1">
    <style>body{background:#111827;color:white;font:14px Arial;margin:12px;width:''' + str(width - 40) + '''px;max-width:calc(100vw - 40px)}.evolution-panel-container{max-width:600px}pre{white-space:pre-wrap}</style>
    <link rel="stylesheet" href="''' + css + '''"></head><body>
    <div id="view-evolution"><div class="evolution-panel-container"></div></div>
    <script>
    const errors=[]; window.addEventListener('error',e=>errors.push(e.message));
    window.calls=[];
    const state={mode:'OFF',status:'stopped',goals:[],resources:{available:true,cpu_percent:3,memory_percent:50},remaining_seconds:0};
    window.fetch=async(url,options={})=>{
      if(options.body){const request=JSON.parse(options.body); calls.push(request);
        if(request.action==='start'){state.mode=request.mode;state.status='running';}
        if(request.action==='pause')state.status='paused';
        if(request.action==='resume')state.status='running';
        if(request.action==='emergency-stop'){state.status='emergency_stopped';state.mode='OFF';state.emergency_latched=true;}
        if(options.headers['X-AXEN-Control']!=='operator')throw Error('missing operator header');
      }
      return new Response(JSON.stringify(state),{headers:{'Content-Type':'application/json'}});
    };
    window.EventSource=class{constructor(){window.feed=this;}close(){}};
    </script><script src="''' + js + '''"></script>
    <pre id="results">pending</pre><script>
    window.addEventListener('load',()=>setTimeout(async()=>{
      const checks=[];const check=(name,pass)=>checks.push({name,pass:!!pass});
      const button=text=>Array.from(document.querySelectorAll('button')).find(b=>b.textContent===text);
      const delay=()=>new Promise(r=>setTimeout(r,30));
      try{
        document.querySelector('input[type=checkbox]').checked=true;
        button('Start learning').click();await delay();check('one start starts worker',calls[0].action==='start'&&calls[0].offline_demo);
        feed.onmessage({data:JSON.stringify({id:1,at:'2026-10-14T00:00:00Z',kind:'source',data:{url:'https://docs.python.org/3/tutorial/controlflow.html',title:'<img src=x onerror=alert(1)>'}})});
        feed.onmessage({data:JSON.stringify({id:2,at:'2026-10-14T00:00:01Z',kind:'evaluation',data:{candidate:{held_out:{passed:3,total:3}},scope:'arithmetic fixture only'}})});
        await delay();check('research source linked',document.querySelector('a').href.includes('docs.python.org'));
        check('untrusted source rendered as text',document.querySelectorAll('img').length===0);
        check('actual event results rendered',document.body.textContent.includes('arithmetic fixture only'));
        button('Pause').click();await delay();check('pause request',calls.at(-1).action==='pause');
        button('Resume').click();await delay();check('resume request',calls.at(-1).action==='resume');
        button('Emergency stop').click();await delay();check('emergency latched',document.body.textContent.includes('Emergency stop: Latched'));
        check('no horizontal overflow',document.documentElement.scrollWidth<=window.innerWidth);
        check('panel fits declared viewport',document.querySelector('.autonomy-panel').getBoundingClientRect().right<=document.body.clientWidth+12);
        check('no JS errors',errors.length===0);
      }catch(error){checks.push({name:String(error),pass:false});}
      document.getElementById('results').textContent=JSON.stringify(checks);
    },100));</script></body></html>'''
    fixture = tmp_path / "dashboard.html"; fixture.write_text(html, encoding="utf-8")
    result = subprocess.run([str(BROWSER), "--headless", "--disable-gpu", "--no-first-run",
        "--force-device-scale-factor=1",
        "--disable-background-networking", "--disable-crash-reporter", "--disable-breakpad",
        "--user-data-dir=" + str(tmp_path / "browser-profile"),
        "--window-size=" + str(width) + "," + str(height), "--virtual-time-budget=2500",
        "--screenshot=" + str(tmp_path / "dashboard.png"), "--dump-dom", fixture.as_uri()],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=45)
    match = re.search(r'<pre id="results"[^>]*>(.*?)</pre>', result.stdout, re.S)
    assert match and match.group(1) != "pending", result.stderr[-1500:]
    checks = json.loads(match.group(1))
    assert all(check["pass"] for check in checks), checks
