from flask import Flask, request, jsonify, make_response
from flask_restplus import Api, Resource, fields
import joblib
import pandas as pd
import numpy as np
from sklearn import preprocessing

flask_app = Flask(__name__)
app = Api(app = flask_app, 
		  version = "1.0", 
		  title = "ML React App", 
		  description = "Predict results using a trained model")

name_space = app.namespace('HeartDiseaseApi', description='Prediction APIs')
name_space1= app.namespace('BreastCancerApi',description='Prediction APIs1')

model1 = app.model('Prediction params', {
				  'BMICategory': fields.String(required = True, 
				  							description="BMI Category", ),
				  'Smoking': fields.String(required = True, 
				  							description="Smoking", 
    					  				 	),
				  'AlcoholDrinking': fields.String(required = True, 
				  							description="Alcohol Drinking", 
    					  				 	),
				  'Stroke': fields.String(required = True, 
				  							description="Stroke", 
    					  				 	),
				  'DiffWalking': fields.String(required = True, 
				  							description="Difficulty in Walking", 
    					  				 	),
				  'Sex': fields.String(required = True, 
				  							description="Sex", 
    					  				 	),
				  'AgeCategory': fields.String(required = True, 
				  							description="Age Category", 
    					  				 	),
				   'Race': fields.String(required = True, 
				  							description="Race", 
    					  				 	),
					'Diabetic': fields.String(required = True, 
				  							description="Diabetic", 
    					  		            ),
					'PhysicalActivity': fields.String(required = True, 
					description="Physical Activity", 
					),
					'GenHealth': fields.String(required = True, 
					description="General Health", 
					),
					'Asthma': fields.String(required = True, 
					description="Asthma", 
					),
					'KidneyDisease': fields.String(required = True, 
					description="Kidney Disease", 
					),
					'SkinCancer': fields.String(required = True, 
					description="Skin Cancer", 
					),})
model2=app.model('Prediction Params',
                {'clump_thickness': fields.Float(required = True, 
				  							   description="Clump Thickness", 
    					  				 	   ),
				  'uniform_cell_size': fields.Float(required = True, 
				  							   description="Uniform Cell Size", 
    					  				 	   ),
				  'uniform_cell_shape': fields.Float(required = True, 
				  							description="Uniform Cell Shape", 
    					  				 	),
				  'marginal_adhesion': fields.Float(required = True, 
				  							description="Marginal Adhesion", 
    					  				 	),
				  'single_epithelial_size': fields.Float(required = True, 
				  							description="Single EpiThelial Size", 
    					  				 	),
				  'bare_nuclei': fields.Float(required = True, 
				  							description="Bare Nuclei", 
    					  				 	),
				  'bland_chromatin': fields.Float(required = True, 
				  							description="Bland Chromatin", 
    					  				 	),
				  'normal_nucleoli': fields.Float(required = True, 
				  							description="Normal Nucleoli", 
    					  				 	),
				  'mitoses': fields.Float(required = True, 
				  							description="Mitoses", 
    					  				 	)


})
clf =joblib.load('clf.joblib')

classifier = joblib.load('XGBnew.pkl')
DATASET_PATH = 'Dataset/Cleaned.csv'

@name_space.route("/")
class MainClass(Resource):

	def options(self):
		response = make_response()
		response.headers.add("Access-Control-Allow-Origin", "*")
		response.headers.add('Access-Control-Allow-Headers', "*")
		response.headers.add('Access-Control-Allow-Methods', "*")
		return response
	

	@app.expect(model1)		
	def post(self):
		try:
			formData = request.json
			input_features = [val for val in formData.values()]
			features_value = [np.array(input_features)]
			features_name = ['BMICategory','Smoking','AlcoholDrinking','Stroke','DiffWalking','Sex','AgeCategory','Race','Diabetic','PhysicalActivity','GenHealth','Asthma','KidneyDisease','SkinCancer']
			input_df = pd.DataFrame(features_value, columns=features_name)
			df = pd.read_csv(DATASET_PATH, index_col=0)
			df=pd.concat([df, input_df], axis=0)
			order_cols=["BMICategory", "AgeCategory"]
			no_order_cols=["Smoking", "AlcoholDrinking", "Stroke", "DiffWalking",
                     "Sex", "Race", "Diabetic", "PhysicalActivity",
                     "GenHealth", "Asthma", "KidneyDisease", "SkinCancer"]
			for col in order_cols:
				df[col]= preprocessing.LabelEncoder().fit_transform(df[col])
			for col in no_order_cols:
				dummy_col = pd.get_dummies(df[col], prefix=col)
				df = pd.concat([df, dummy_col], axis=1)
				del df[col]
			df.drop('HeartDisease', axis=1, inplace=True)
			output = classifier.predict_proba(df[-1:])
			pred = round(output[0][1]*100, 2)
			prediction_text='Patient has {0}% chances of having cancer.'.format(pred)
			response = jsonify({
				"statusCode": 200,
				"status": "Prediction made",
				"result":  prediction_text,
				})
			response.headers.add('Access-Control-Allow-Origin', '*')
			return response
		except Exception as error:
			return jsonify({
				"statusCode": 500,
				"status": "Could not make prediction",
				"error": str(error)
			})
@name_space1.route("/")
class MainClass(Resource):

	def options(self):
		response = make_response()
		response.headers.add("Access-Control-Allow-Origin", "*")
		response.headers.add('Access-Control-Allow-Headers', "*")
		response.headers.add('Access-Control-Allow-Methods', "*")
		return response
	@app.expect(model2)
	def post(self):
		try:
			formData=request.json
			input_features = [int(x) for x in formData.values()]
			features_value = [np.array(input_features)]
			features_name = ['clump_thickness', 'uniform_cell_size', 'uniform_cell_shape', 'marginal_adhesion',
							'single_epithelial_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses']
			df = pd.DataFrame(features_value, columns=features_name)
			output = clf.predict(df)
			if output == 4:
				res_val = "a high risk of Breast Cancer"
			else:
				res_val = "a low risk of Breast Cancer"
				prediction_text='Patient has {0}'.format(res_val)
				response = jsonify({
						"statusCode": 200,
						"status": "Prediction made",
						"result":  prediction_text,
						})
				response.headers.add('Access-Control-Allow-Origin', '*')
				return response
		except Exception as error:
			return jsonify({
				"statusCode": 500,
				"status": "Could not make prediction",
				"error": str(error)
			})
