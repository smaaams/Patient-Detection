import json
import pickle
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB 
from sklearn.preprocessing import OneHotEncoder


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
        elif classifier_type == 6:
            self.model = self.build_naive_bayes_classifier()
        # self.model = LogisticRegression(n_jobs=-1)
        # self.model = SVC(class_weight='balanced')
        # self.model = SGDClassifier(class_weight='balanced', n_jobs=-1)
        # self.model = DecisionTreeClassifier(class_weight='balanced')
        # self.model = RandomForestClassifier(n_estimators=51, class_weight='balanced', n_jobs=-1)
        # self.model = AdaBoostClassifier(n_estimators=201)
        # self.model = MultinomialNB()

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
    
    def build_naive_bayes_classifier(self):
        return MultinomialNB()

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

    def predict(self, reviews):
        feature_vectors = self.vectorizer.transform(reviews)
        return self.model.predict(feature_vectors)


class DrugApproximator:
    def __init__(self, classifier_type):
        # TODO: for each classifier, define path where they are going to be save, like below
        """"
        self.logistic_regression_path = <MODEL PATH>
        """
        with open('data/drugvectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)


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
        elif classifier_type == 6:
            self.model = self.build_naive_bayes_classifier()

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
    
    def build_naive_bayes_classifier(self):
        return MultinomialNB()

    def one_hot_encoder(self, conditions):
        condition_encoding = np.zeros((len(conditions),len(self.condition_list)))
        for i, condition in enumerate(conditions):
            condition_encoding[i][-(self.condition_list.index(condition))] = 1
        return(condition_encoding)


    def train(self, data_set):
        reviews = [ review for review, _, _ in data_set ]
        conditions = [condition for _, condition, _ in data_set]
        drugs = [drug for _, _, drug in data_set]
        self.condition_list = list(set(conditions))
        feature_vectors = self.vectorizer.fit_transform(reviews).toarray()
        condition_encoding = self.one_hot_encoder(conditions)
        feature_vectors = np.concatenate((feature_vectors, condition_encoding), axis = 1)
        feature_vectors = csr_matrix(feature_vectors)

        self.model.fit(feature_vectors, drugs)


    def evaluate(self, data_set):
        reviews = [review for review, _, _ in data_set]
        conditions = [condition for _, condition, _ in data_set]
        drugs = [drug for _, _, drug in data_set]

        feature_vectors = self.vectorizer.transform(reviews).toarray()

        condition_encoding = self.one_hot_encoder(conditions)
        feature_vectors = np.concatenate((feature_vectors, condition_encoding), axis = 1)
        feature_vectors = csr_matrix(feature_vectors)

        predictions = self.model.predict(feature_vectors)
        report = classification_report(drugs, predictions)
        print(report)

        confusion_mat = confusion_matrix(drugs, predictions)
        print(confusion_mat)

    def approximate(self, reviews, patient_detector):
        predicted_conditions = patient_detector.predict(reviews)
        feature_vectors = self.vectorizer.transform(reviews).toarray()
        condition_encoding = self.one_hot_encoder(predicted_conditions)
        feature_vectors = np.concatenate((feature_vectors, condition_encoding), axis = 1)
        feature_vectors = csr_matrix(feature_vectors)

        drug_predictions = self.model.predict(feature_vectors)

        return drug_predictions



if __name__ == '__main__':
    with open('data/train.json', 'r') as json_file:
        data_points = json.load(json_file)
        train_set = [(data_point['review'], data_point['condition']) for data_point in data_points]
    with open('data/test.json', 'r') as json_file:
        data_points = json.load(json_file)
        test_set = [(data_point['review'], data_point['condition']) for data_point in data_points]

    patient_detector = PatientDetector(0)
    patient_detector.train(train_set)
    patient_detector.evaluate(test_set)

    with open('data/vectorizer.pkl', 'wb') as pickle_file:
        pickle.dump(patient_detector.vectorizer, pickle_file)
    with open('data/model.pkl', 'wb') as pickle_file:
        pickle.dump(patient_detector.model, pickle_file)

    with open('data/dtrain.json', 'r') as json_file:
        data_points = json.load(json_file)
        train_set = [(data_point['review'], data_point['condition'], data_point['drug']) for data_point in data_points]
    with open('data/dtest.json', 'r') as json_file:
        data_points = json.load(json_file)
        test_set = [(data_point['review'], data_point['condition'], data_point['drug']) for data_point in data_points]

    drug_approximator = DrugApproximator(0)
    drug_approximator.train(train_set)
    drug_approximator.evaluate(test_set)

    #approximated_drug = drug_approximator.approximate(['it has no side effect i take it in combination of bystolic mg and fish oil'], patient_detector)
    #print(approximated_drug)


