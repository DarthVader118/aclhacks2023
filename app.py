from flask import Flask, render_template, request
import subprocess

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/execute_opencv_file', methods=['GET'])
def execute_opencv_file():
    try:
        subprocess.run(['python', 'opencv_file.py'], check=True)
        return 'OpenCV Python file executed successfully'
    except subprocess.CalledProcessError as e:
        return f'Error executing OpenCV Python file: {str(e)}'

if __name__ == '__main__':
    app.run(use_reloader=True, debug=True)