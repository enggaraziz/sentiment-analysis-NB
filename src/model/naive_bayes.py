import os
import json
import pickle

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score
)
from sklearn.model_selection import cross_val_score
from plotly import figure_factory as ff

class NaiveBayesClassifier:
    def __init__(self, save_path) -> None:
        self.filename = "/naive_bayes_model.bin"
        self.cf_matrix_fn = "/{}_cf_matrix.png"
        self.metrics_fn = "/{}_metrics_score.json"
        self.model_path = save_path+self.filename
        self.cf_matrix_png_path = save_path+self.cf_matrix_fn
        self.metrics_json_path = save_path+self.metrics_fn
        self.classifier = MultinomialNB()
        self.training_performance = {}
        self.validation_performance = {}
        self.cross_val = 0
        self.labels = [0, 1]

    def get_performances(self, datas, pred_result, is_train=False):
        data_type = "train" if is_train else "validation"
        accuracy = accuracy_score(datas[data_type+'_label'], pred_result)
        f1 = f1_score(datas[data_type+'_label'], pred_result)
        precision = precision_score(datas[data_type+'_label'], pred_result)
        recall = recall_score(datas[data_type+'_label'], pred_result)
        total_data = len(pred_result)
        cf_matrix = confusion_matrix(datas[data_type+'_label'], pred_result, labels=self.labels)
        performances = {
            "total_data": total_data,
            "cf_matrix": cf_matrix.tolist(),
            "f1": round(f1*100, 2),
            "accuracy": round(accuracy*100, 2),
            "precision": round(precision*100, 2),
            "recal": round(recall*100, 2)
        }

        if is_train:
            cross_val = cross_val_score(
                self.classifier,
                datas[data_type+'_data'],
                datas[data_type+'_label']
                )
            cross_val = [round(v*100, 2) for v in cross_val]
            performances['cross_val'] = cross_val

        return performances

    def train(self, training_data: dict):
        self.model = self.classifier.fit(
            training_data['train_data'],
            training_data['train_label']
        )
        validation_result = self.model.predict(training_data['validation_data'])
        self.validation_performance = self.get_performances(training_data, validation_result)

        training_result = self.model.predict(training_data['train_data'])
        self.training_performance = self.get_performances(training_data, training_result, is_train=True)

        self._save_model()
        self._save_performance()

    def _save_performance(self):
        # save confussion matrix
        self._save_performance_data(self.validation_performance)
        self._save_performance_data(self.training_performance, is_train=True)
        self._save_fig(self.validation_performance['cf_matrix'])
        self._save_fig(self.training_performance['cf_matrix'], is_train=True)

    def _save_performance_data(self, data, is_train=False):
        dtype = 'train' if is_train else 'validation'
        data = {k: v for k, v in data.items()}

        with open(self.metrics_json_path.format(dtype), 'w') as writter:
            json.dump(data, writter)


    def _save_fig(self, data, is_train=False):
        title = 'Confussion matrix {} data'.format("training" if is_train else "validation") 
        
        data = data[::-1]
        x_axis_ = ['negative', 'positive']
        y_axis_ = x_axis_[::-1].copy()
        data_str = [[str(y) for y in x] for x in data]

        # set up figure 
        fig = ff.create_annotated_heatmap(data, x=x_axis_, y=y_axis_, annotation_text=data_str, colorscale='Cividis')

        # add title
        fig.update_layout(title_text=f'<b>{title}</b>')

        # set transparent bg
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')

        # add colorbar
        fig['data'][0]['showscale'] = True
        fn = self.cf_matrix_png_path.format('train' if is_train else 'validation')
        fig.write_image(fn,format='png')

    def _save_model(self):
        with open(self.model_path, 'wb') as writter:
            pickle.dump(self, writter)

    def predict(self, data):
        predictions = {}
        predictions['sentiment'] = self.model.predict(data)
        predictions['probability'] = self.model.predict_proba(data)
        return predictions
