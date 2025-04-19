import re
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def show_logs():
    log_lines = []
    pattern = r"\[\d{4}-\d{2}-\d{2}[\s,:\d]+\]\s*\[[^\]]+\]\s*(.*)"

    with open('task_log-2.log', 'r') as f: #CHANGE PATH!!!!
        for idx, line in enumerate(f):
            if idx == 0:
                continue  # skip the first line
            match = re.match(pattern, line)
            if match:
                log_lines.append(match.group(1).strip())
            else:
                log_lines.append(line.strip())  # fallback if pattern doesn't match

    return render_template('dashboard.html', logs=log_lines)

if __name__ == '__main__':
    app.run(debug=True)
