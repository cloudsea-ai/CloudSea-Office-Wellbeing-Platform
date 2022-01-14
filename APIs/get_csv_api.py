from flask import Flask, send_from_directory

app = Flask(__name__)
# Specify directory to download from . . .
DOWNLOAD_DIRECTORY = "/home/cloweews/Projects/CloudSea-Office-Wellbeing-Platform/results"

@app.route('/getresults',methods = ['GET'])
def get_files():

    """Download a file."""
    try:
        return send_from_directory(DOWNLOAD_DIRECTORY, 'results_table.csv', as_attachment=True)
    except FileNotFoundError:
        abort(404)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 5001, threaded = True, debug = True)
