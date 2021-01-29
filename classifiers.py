import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from LP_toolkits import normalizer
import numpy as np


class PatientDetector:
    def __init__(self, classifier_type):
        # TODO: for each classifier, define path where they are going to be save, like below
        """"
        self.logistic_regression_path = <MODEL PATH>
        """
        self.vectorizer = TfidfVectorizer()
        self.feature_selector = SelectKBest(chi2, k=1000)

        if classifier_type == 0:
            self.model = self.build_logistic_regression_classifier()
        elif classifier_type == 1:
            self.model = self.build_svc_classifier()
        elif classifier_type == 2:
            self.model = self.build_sgd_classifier()
        elif classifier_type == 3:
            self.model = self.build_decision_tree_classifier()
        elif classifier_type == 4:
            self.model = self.build_random_forest_classifier()
        elif classifier_type == 5:
            self.model = self.build_adaboost_classifier()
        # self.model = LogisticRegression(n_jobs=-1)
        # self.model = SVC(class_weight='balanced')
        # self.model = SGDClassifier(class_weight='balanced', n_jobs=-1)
        # self.model = DecisionTreeClassifier(class_weight='balanced')
        # self.model = RandomForestClassifier(n_estimators=51, class_weight='balanced', n_jobs=-1)
        # self.model = AdaBoostClassifier(n_estimators=201)

    def build_logistic_regression_classifier(self):
        return LogisticRegression(n_jobs=-1)

    def build_svc_classifier(self):
        return SVC(class_weight='balanced')

    def build_sgd_classifier(self):
        return SGDClassifier(class_weight='balanced', n_jobs=-1)

    def build_decision_tree_classifier(self):
        return DecisionTreeClassifier(class_weight='balanced')

    def build_random_forest_classifier(self):
        return RandomForestClassifier(n_estimators=51, class_weight='balanced', n_jobs=-1)

    def build_adaboost_classifier(self):
        return AdaBoostClassifier(n_estimators=201)

    def train(self, data_set):
        reviews = [review for review, _ in data_set]
        conditions = [condition for _, condition in data_set]

        feature_vectors = self.vectorizer.fit_transform(reviews)
        # feature_vectors = self.feature_selector.fit_transform(feature_vectors, conditions)

        self.model.fit(feature_vectors, conditions)

    def evaluate(self, data_set):
        reviews = [review for review, _ in data_set]
        conditions = [condition for _, condition in data_set]

        feature_vectors = self.vectorizer.transform(reviews)
        # feature_vectors = self.feature_selector.transform(feature_vectors)

        predictions = self.model.predict(feature_vectors)

        report = classification_report(conditions, predictions)
        print(report)

        confusion_mat = confusion_matrix(conditions, predictions)
        print(confusion_mat)

    def query(self, review):
        # TODO: save each classifier model to be able to load them here, not train them again!!
        return self.model.predict(np.array([normalizer(review)]))


if __name__ == '__main__':
    with open('data/train.json', 'r') as json_file:
        data_points = json.load(json_file)
        train_set = [(data_point['review'], data_point['condition']) for data_point in data_points]
    with open('data/test.json', 'r') as json_file:
        data_points = json.load(json_file)
        test_set = [(data_point['review'], data_point['condition']) for data_point in data_points]

    patient_detector = PatientDetector(1)
    patient_detector.train(train_set)
    patient_detector.evaluate(test_set)
