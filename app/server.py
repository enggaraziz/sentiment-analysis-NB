import os
import sys
import pandas as pd
os.environ["PARRENT_PATH"] = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ["PARRENT_PATH"])
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import (
    Flask,
    render_template,
    request,
    Response,
    redirect
)

# load twitter credentials
from dotenv import load_dotenv
load_dotenv()

# load internal module
from src.wrapper import train
from app.app_util import (
    read_performance_data,
    read_tweet_analysis_data,
    get_tweet_data,
    check_model_exist,
    check_tweet_doc_exist
)


app = Flask(__name__)
app.secret_key = "secret"

# model location path
TRAIN_UPLOAD_FOLDER = os.environ["PARRENT_PATH"] + '/app/files/train-data/'
os.environ['TRAIN_UPLOAD_FOLDER'] = TRAIN_UPLOAD_FOLDER
MODEL_LOG_PATH = os.environ["PARRENT_PATH"] + '/app/log/model_list.csv'
os.environ['MODEL_LOG_PATH'] = MODEL_LOG_PATH
MODEL_PATH = os.environ["PARRENT_PATH"] + '/app/static/models/'
os.environ['MODEL_PATH'] = MODEL_PATH

# tweet location path
TWEETS_DOCS_PATH = os.environ["PARRENT_PATH"] + '/app/files/tweets/'
os.environ['TWEETS_DOCS_PATH'] = TWEETS_DOCS_PATH
TWEETS_LOG_PATH = os.environ["PARRENT_PATH"] + '/app/log/tweet_doc_list.csv'
os.environ['TWEETS_LOG_PATH'] = TWEETS_LOG_PATH

# allowed file
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def _write_log(request: request, date: datetime, result, save_path):
    # model_name,ratio,created_at,dataset_name,status
    df = pd.DataFrame.from_dict(
        {
            "model_name": [request.form['model-name'].replace(' ', '-')],
            "ratio": [float(request.form['ratio'])],
            "created_at": ["{}-{}-{} {}:{}:{}".format(
                date.year,
                date.month,
                date.day,
                date.hour,
                date.minute,
                date.second)],
            "dataset_file": [request.files['dataset'].filename],
            "save_path": ['/'.join(save_path.split('/')[-2:])],
            "status": ["Finished"]

        })
    df.to_csv(MODEL_LOG_PATH, mode="a", index=False, header=not os.path.exists(MODEL_LOG_PATH))

@app.route('/trainer', methods=['GET', 'POST'])
def trainer():
    if request.method == 'POST':
        file = request.files['dataset']
        curr_date = datetime.now()
        model_name = request.form["model-name"].replace(' ','-')
        folder_name = (
            '{}_{}{}{}{}{}'.format(
                model_name,
                curr_date.month,
                curr_date.day,
                curr_date.hour,
                curr_date.minute,
                curr_date.second
        ))
        save_path = MODEL_PATH+folder_name
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_path = os.path.join(TRAIN_UPLOAD_FOLDER, filename)
            file.save(upload_path)
            try:
                result = train(upload_path, float(request.form['ratio']), save_path)
            except Exception as e:
                return Response("{'message': '" + str(e) + "'}", status=500, mimetype="application/json")
            _write_log(request, curr_date, result, save_path)
            return render_template("model-performances.html")

@app.route('/')
def dashboard_page():
    return render_template('dashboard.html')


@app.route('/train-model')
def train_model_page():
    return render_template('train-model.html')


@app.route('/model-performances')
def model_performance_page():
    models = check_model_exist(MODEL_LOG_PATH)
    if models:
        # load list model
        df = pd.read_csv(MODEL_LOG_PATH)
        return render_template(
            'model-performances.html',
            col_name=df.columns.values,
            row_data=list(df.values.tolist()),
            zip=zip)
    return redirect('/empty-model')

@app.route('/performance/<string:id>')
def performance(id):
    # load list model
    model_path = MODEL_PATH+id+'/'
    performance_data = read_performance_data(model_path)
    return render_template(
        'performances.html',
        id=id,
        performance_data=performance_data,
        train_png_cf_path=f'models/{id}/train_cf_matrix.png',
        validation_png_cf_path=f'models/{id}/validation_cf_matrix.png'
        )

@app.route('/analyze-tweet', methods=['GET','POST'])
def analyze_tweet():
    if request.method == 'POST':
        try:
            tweets = get_tweet_data(
                request.form['model-name'],
                request.form['keywords'],
                request.form.getlist("tweet-filters"),
                request.form['tweet-types'],
                total_data=request.form['total-data'],
                exclude_filter=True if 'exclude-mode' in request.form.keys() else False
            )
            if tweets:
                return Response("OK", status=200, mimetype="application/json")
            return Response("There is no data that we retrieved, please use another query", status=500, mimetype="application/json")
        except Exception as error:
            return Response("error: " + str(error), status=500, mimetype="application/json")


@app.route('/empty-model')
def empty_model():
    return render_template('empty-model.html')


@app.route('/sentiment-analysis')
def sentiment_analysis_page():
    # check wether any model exist
    models = check_model_exist(MODEL_LOG_PATH)
    if models:
        tweets = check_tweet_doc_exist(TWEETS_LOG_PATH)
        if tweets:
            df = pd.read_csv(TWEETS_LOG_PATH)
            return render_template(
                'sentiment-analysis.html',
                list_models=models,
                tweet_doc_exist=True,
                col_name=df.columns.values,
                row_data=list(df.values.tolist()),
                zip=zip
                )
        return render_template('sentiment-analysis.html', list_models=models, tweet_doc_exist=False)
    return redirect('/empty-model')

@app.route('/analyze-tweet/<string:id>')
def insight(id):
    # load list model
    tweet_docs_path = TWEETS_DOCS_PATH+id
    insight_data, tweet_df = read_tweet_analysis_data(tweet_docs_path)
    return render_template(
        'tweet-insights.html',
        id=id,
        insight_data=insight_data,
        col_name=tweet_df.columns.values,
        row_data=list(tweet_df.values.tolist()),
        zip=zip
        )

# @app.route('')


if __name__ == '__main__':
    app.run(host="localhost",debug=True)
