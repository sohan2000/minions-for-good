from flask import Flask, render_template, request

app = Flask(__name__)

# These would normally come from your database or model
home_description_text = ""
predicted_date = "December 15, 2025"
progress_percent = 45  # update dynamically later

# Move highlighted dates to Python
highlighted_dates = [
    "2025-04-10",
    "2025-04-15",
    "2025-04-27"
]

@app.route('/', methods=['GET', 'POST'])
def dashboard():
    global home_description_text

    if request.method == 'POST':
        home_description_text = request.form.get('home_description')
        print("Submitted description:", home_description_text)

    return render_template('dashboard.html',
                           predicted_date=predicted_date,
                           progress_percent=progress_percent,
                           highlighted_dates=highlighted_dates)

if __name__ == '__main__':
    app.run(debug=True)
