# serve_lesson.py
# Simple Flask server to show final_lesson_plan.json in the browser with an embedded quiz (no modal dialogs).
# Usage: python serve_lesson.py
from flask import Flask, jsonify, send_file, send_from_directory, request
from pathlib import Path
import json, webbrowser, threading, time, os

BASE_DIR = Path(__file__).resolve().parent
LESSON_FILE = BASE_DIR / "final_lesson_plan.json"
HOST = "127.0.0.1"
PORT = int(os.environ.get("LESSON_PORT", 7860))

app = Flask(__name__, static_folder=str(BASE_DIR / "static"), static_url_path="/static")

@app.route("/api/lesson")
def api_lesson():
    if not LESSON_FILE.exists():
        return jsonify({"error": "final_lesson_plan.json not found"}), 404
    try:
        with open(LESSON_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": "failed to read lesson file", "detail": str(e)}), 500

@app.route("/")
def index():
    # Inline simple HTML/JS to avoid extra files
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Lesson Viewer</title>
  <style>
    body {font-family: Arial, sans-serif; margin: 20px; max-width: 900px;}
    h1 {font-size: 1.6rem;}
    .bullet {margin: 6px 0;}
    .quiz q {display:block; margin: 14px 0; padding: 8px; border-radius:6px; background:#f6f6f6;}
    .choice {margin:6px 0;}
    button {padding:8px 12px; margin-top:10px;}
    #result {margin-top:12px; font-weight:600;}
    .small {color:#666; font-size:0.9rem;}
  </style>
</head>
<body>
  <h1 id="moduleTitle">Lesson</h1>
  <div id="explanation"></div>
  <hr/>
  <h2>Quiz</h2>
  <div id="quizArea">Loading quiz...</div>
  <div id="controls"></div>
  <div id="result"></div>
  <script>
    async function fetchLesson() {
      const res = await fetch('/api/lesson');
      if (!res.ok) {
        document.getElementById('explanation').innerText = "Lesson file not found or error. Run the pipeline first.";
        document.getElementById('quizArea').innerText = "";
        return null;
      }
      return res.json();
    }

    function renderExplanation(expl){
      const container = document.getElementById('explanation');
      container.innerHTML = "";
      if(!expl) return;
      const title = expl.title || "";
      if(title) {
        const h = document.createElement('h2');
        h.innerText = title;
        container.appendChild(h);
      }
      const bullets = expl.bullets || expl.points || [];
      if(Array.isArray(bullets) && bullets.length){
        bullets.slice(0, 50).forEach(b=>{
          const d = document.createElement('div');
          d.className = 'bullet';
          if(typeof b === 'string') d.innerText = "• " + b;
          else d.innerText = "• " + (b.title || b.content || JSON.stringify(b));
          container.appendChild(d);
        });
      } else if (typeof expl === 'string'){
        container.innerText = expl;
      } else {
        container.innerText = "No explanation text available.";
      }
    }

    function renderQuiz(quiz){
      const qa = document.getElementById('quizArea');
      qa.innerHTML = "";
      if(!Array.isArray(quiz) || quiz.length===0){
        qa.innerText = "No quiz available.";
        return;
      }
      // build DOM for each question
      quiz.forEach((q, qi) => {
        const qdiv = document.createElement('div');
        qdiv.className = 'q';
        const qtext = document.createElement('q');
        qtext.innerText = (q.question || q.questionText || 'Question');
        qdiv.appendChild(qtext);

        const choices = q.choices;
        if(Array.isArray(choices) && choices.length){
          choices.forEach((c, ci) => {
            const label = document.createElement('label');
            label.className = 'choice';
            const input = document.createElement('input');
            input.type = 'radio';
            input.name = 'q_'+qi;
            input.value = ci;
            label.appendChild(input);
            const txt = document.createTextNode(' ' + (typeof c === 'string' ? c : (c.title||c.content||JSON.stringify(c))));
            label.appendChild(txt);
            qdiv.appendChild(label);
          });
        } else {
          // open question: textarea
          const ta = document.createElement('textarea');
          ta.name = 'q_'+qi;
          ta.rows = 3;
          ta.style.width = '100%';
          qdiv.appendChild(ta);
        }
        qa.appendChild(qdiv);
      });

      // submit button
      const controls = document.getElementById('controls');
      controls.innerHTML = '';
      const submit = document.createElement('button');
      submit.innerText = 'Submit Quiz';
      submit.onclick = ()=> gradeQuiz(quiz);
      controls.appendChild(submit);

      const exportBtn = document.createElement('button');
      exportBtn.innerText = 'Export Results (JSON)';
      exportBtn.style.marginLeft = '8px';
      exportBtn.onclick = ()=> {
        const res = window.latestResult || {};
        const blob = new Blob([JSON.stringify(res, null, 2)], {type: 'application/json'});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = 'quiz_results.json'; a.click();
      };
      controls.appendChild(exportBtn);
    }

    function gradeQuiz(quiz){
      const results = [];
      let correct=0;
      quiz.forEach((q, qi) => {
        const name = 'q_'+qi;
        let given = null;
        // radio?
        const radios = document.getElementsByName(name);
        if(radios && radios.length && radios[0].type === 'radio'){
          for(const r of radios) if(r.checked) { given = r.value; break; }
        } else {
          const ta = document.getElementsByName(name)[0];
          if(ta) given = ta.value;
        }
        const expected = q.correct_index != null ? String(q.correct_index) : (q.correct || null);
        let ok = false;
        if(expected != null){
          if(expected.toString() === (given==null?'' : given.toString())) ok = true;
        }
        if(ok) correct++;
        results.push({id: q.id || qi, expected: expected, given: given, ok: ok});
      });
      const score = (correct / Math.max(1, quiz.length));
      const summary = {score: score, correct: correct, total: quiz.length, details: results};
      window.latestResult = summary;
      document.getElementById('result').innerText = `Score: ${(score*100).toFixed(0)}% (${correct}/${quiz.length})`;
    }

    (async function(){
      const lesson = await fetchLesson();
      if(!lesson) return;
      document.getElementById('moduleTitle').innerText = lesson.moduleTitle || lesson.explanation?.title || 'Lesson';
      renderExplanation(lesson.explanation || lesson);
      renderQuiz(lesson.quiz || lesson.questions || []);
      // quick feedback in console
      console.log('Lesson loaded in browser; quiz ready.');
    })();
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    # start server in main thread; open browser after short delay
    def open_browser_delayed():
        # wait briefly to let Flask start
        time.sleep(0.6)
        url = f"http://{HOST}:{PORT}/"
        try:
            webbrowser.open(url)
            print("Opened browser at", url)
        except Exception:
            print("Please open your browser and visit", url)

    print(f"Starting Lesson server at http://{HOST}:{PORT}/ (serving {LESSON_FILE})")
    threading.Thread(target=open_browser_delayed, daemon=True).start()
    app.run(host=HOST, port=PORT, debug=False)
