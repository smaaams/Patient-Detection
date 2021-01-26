import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


class PatientDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.feature_selector = SelectKBest(chi2, k=1000)
        # self.model = LogisticRegression(n_jobs=-1)
        # self.model = SVC(class_weight='balanced')
        # self.model = SGDClassifier(class_weight='balanced', n_jobs=-1)
        # self.model = DecisionTreeClassifier(class_weight='balanced')
        self.model = RandomForestClassifier(n_estimators=51, class_weight='balanced', n_jobs=-1)
        # self.model = AdaBoostClassifier(n_estimators=201)

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


if __name__ == '__main__':
    with open('data/train.json', 'r') as json_file:
        data_points = json.load(json_file)
        train_set = [(data_point['review'], data_point['condition']) for data_point in data_points]
    with open('data/test.json', 'r') as json_file:
        data_points = json.load(json_file)
        test_set = [(data_point['review'], data_point['condition']) for data_point in data_points]

    patient_detector = PatientDetector()
    patient_detector.train(train_set)
    patient_detector.evaluate(test_set)
