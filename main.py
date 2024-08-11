from flask import Flask, request, make_response
import pandas as pd
import numpy as np
import io

import requests

app = Flask(__name__)

def humanize_data(data):
    # Add some noise and slight variations to make data appear more human-like
    noise = np.random.normal(0, 0.01, len(data))
    data += noise
    return data

@app.route('/generate_csv_normal_std', methods=['GET'])
def generate_csv_normal_std():
    try:
        mean = float(request.args.get('mean', 0))
        std = float(request.args.get('std', 1))
        num_samples = int(request.args.get('num_samples', 100))

        # Generate normal distribution data
        data = np.random.normal(mean, std, num_samples)

        # Humanize the data
        data = humanize_data(data)

        # Create a DataFrame
        df = pd.DataFrame({'value': data})

        # Convert DataFrame to CSV
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        # Create the response
        response = make_response(output.getvalue())
        response.headers['Content-Disposition'] = 'attachment; filename=generated_data_normal_std.csv'
        response.headers['Content-Type'] = 'text/csv'

        return response
    except Exception as e:
        return str(e), 400

@app.route('/generate_csv_normal_minmax', methods=['GET'])
def generate_csv_normal_minmax():
    try:
        mean = float(request.args.get('mean', 0))
        minimum = float(request.args.get('min', mean - 1))
        maximum = float(request.args.get('max', mean + 1))
        num_samples = int(request.args.get('num_samples', 100))

        if minimum >= maximum:
            return "Error: 'min' must be less than 'max'", 400

        # Approximate standard deviation
        std = (maximum - minimum) / 6

        # Generate normal distribution data
        data = np.random.normal(mean, std, num_samples)

        # Truncate data to fit within the min and max range
        data = np.clip(data, minimum, maximum)

        # Humanize the data
        data = humanize_data(data)

        # Create a DataFrame
        df = pd.DataFrame({'value': data})

        # Convert DataFrame to CSV
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        # Create the response
        response = make_response(output.getvalue())
        response.headers['Content-Disposition'] = 'attachment; filename=generated_data_normal_minmax.csv'
        response.headers['Content-Type'] = 'text/csv'

        return response
    except Exception as e:
        return str(e), 400

def getRandomNames(num_of_names, first_names, surnames):

  surnames_sample = surnames.sample(n=num_of_names, replace=True).reset_index(drop=True)
  first_names_sample = first_names.sample(n=num_of_names, replace=True).reset_index(drop=True)
  result = pd.concat([first_names_sample, surnames_sample], axis=1)
  result.columns = ['first_name', 'sex', 'surname']
  result = result[['first_name', 'surname', 'sex']]
  
  return result


def getRandomFirstName(num_of_names, gender, first_names):
  first_names_cpy = first_names[first_names['sex'] == gender] if gender else first_names['name']
  first_names_sample = first_names_cpy.sample(n=num_of_names, replace=True).reset_index(drop=True)

  return first_names_sample

def getRandomSecondName(num_of_names, surnames):
  surnames_sample = surnames.sample(n=num_of_names, replace=True).reset_index(drop=True)
  surnames_sample = surnames_sample.str.title()
  return surnames_sample

@app.route('/generate_csv_names', methods=["GET"])
def generate_csv_names():
    try:
        gender = request.args.get('gender')
        num_samples = int(request.args.get("num_samples"))
        first_name = request.args.get("first_name")
        surname = request.args.get("surname")

        first_names = pd.read_csv('https://raw.githubusercontent.com/hadley/data-baby-names/master/baby-names.csv')
        first_names = first_names.drop(['year', 'percent'], axis=1)
        first_names['sex'] = first_names['sex'].replace({'boy': 'male', 'girl': 'female'})
        first_names.drop_duplicates(inplace=True)

        surnames = pd.read_csv('https://raw.githubusercontent.com/fivethirtyeight/data/master/most-common-name/surnames.csv')
        surnames = surnames['name'].str.title()
        surnames.drop_duplicates(inplace=True)
        
        result = pd.DataFrame([])

        # there has to be a better way to do this
        if first_name == "True" and surname == "True":
            result = getRandomNames(num_samples, first_names, surnames)

            if gender:
                result = result[result['sex']==gender]
        
        elif first_name == "True":
            result = getRandomFirstName(num_samples, gender, first_names) if gender else  getRandomFirstName(num_samples, None, first_names)

        else:
            result = getRandomSecondName(num_samples, surnames)

        # Convert DataFrame to CSV
        output = io.StringIO()
        result.to_csv(output, index=False)
        output.seek(0)

        # Create the response
        response = make_response(output.getvalue())
        response.headers['Content-Disposition'] = 'attachment; filename=generated_data_normal_minmax.csv'
        response.headers['Content-Type'] = 'text/csv'

        return response
    except Exception as e:
        return str(e), 400

@app.route('/generate_csv_uniform_minmax', methods=['GET'])
def generate_csv_uniform_minmax():
    try:
        minimum = float(request.args.get('min', 0))
        maximum = float(request.args.get('max', 1))
        num_samples = int(request.args.get('num_samples', 100))

        if minimum >= maximum:
            return "Error: 'min' must be less than 'max'", 400

        # Generate uniform distribution data
        data = np.random.uniform(low=minimum, high=maximum, size=num_samples)

        # Humanize the data
        data = humanize_data(data)

        # Create a DataFrame
        df = pd.DataFrame({'value': data})

        # Convert DataFrame to CSV
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        # Create the response
        response = make_response(output.getvalue())
        response.headers['Content-Disposition'] = 'attachment; filename=generated_data_uniform_minmax.csv'
        response.headers['Content-Type'] = 'text/csv'

        return response
    except Exception as e:
        return str(e), 400





if __name__ == '__main__':
    app.run(debug=True)






# TESTING:  curl -o generated_data_normal_std.csv "http://127.0.0.1:5000/generate_csv_normal_std?mean=10&std=5&num_samples=1000"
# TESTING:  curl -o generated_data_normal_minmax.csv "http://127.0.0.1:5000/generate_csv_normal_minmax?mean=10&min=5&max=15&num_samples=1000"

# change out params for gender, samples, first_name, surname
# TESTING: curl -o generated_data_names.csv "http://127.0.0.1:5000/generate_csv_names?gender=male&num_samples=20&first_name=True&surname=True"

