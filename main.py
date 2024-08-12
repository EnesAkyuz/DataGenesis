from flask import Flask, request, make_response
import pandas as pd
import numpy as np
import io

import requests

app = Flask(__name__)

def humanize_data(data):
    """
​    This function adds some noise and slight variations to the input data to make it appear more human-like.
​
​    Parameters:
​    data (numpy.ndarray): A 1D numpy array containing the input data.
​
​    Returns:
​    numpy.ndarray: A 1D numpy array with the same shape as the input data, but with added noise.
​    """

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
    """
    This function generates a CSV file containing random data from a normal distribution,
    with the mean specified by the user. The data is then truncated
    to fit within a specified minimum and maximum range.

    Parameters:
    mean (float): The mean of the normal distribution. Default is 0.
    min (float): The minimum value for the generated data. Default is mean - 1.
    max (float): The maximum value for the generated data. Default is mean + 1.
    num_samples (int): The number of samples to generate. Default is 100.

    Returns:
    flask.Response: A CSV file containing the generated data, with a filename of 'generated_data_normal_minmax.csv'.
    If an error occurs, returns a string describing the error and a 400 status code.
    """
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
        # data = humanize_data(data)

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
    """
    This function generates a DataFrame containing random names and their corresponding genders.

    Parameters:
    num_of_names (int): The number of random names to generate.
    first_names (pandas.DataFrame): A DataFrame containing a list of first names and their genders.
    surnames (pandas.Series): A Series containing a list of surnames.

    Returns:
    pandas.DataFrame: A DataFrame with three columns: 'first_name', 'sex', and 'surname'. Each row represents a random name.
    """

    surnames_sample = surnames.sample(n=num_of_names, replace=True).reset_index(drop=True)
    first_names_sample = first_names.sample(n=num_of_names, replace=True).reset_index(drop=True)
    result = pd.concat([first_names_sample, surnames_sample], axis=1)
    result.columns = ['first_name', 'sex', 'surname']
    result = result[['first_name', 'surname', 'sex']]

    return result


def getRandomFirstName(num_of_names, gender, first_names):
    """
    This function generates a random sample of first names based on the given gender.

    Parameters:
    num_of_names (int): The number of random first names to generate.
    gender (str, optional): The gender of the first names to generate. If not provided, names of any gender can be generated.
    first_names (pandas.DataFrame): A DataFrame containing a list of first names and their genders.

    Returns:
    pandas.Series: A Series containing the randomly sampled first names. The index is reset to a range index.
    """
    first_names_cpy = first_names[first_names['sex'] == gender] if gender else first_names['name']
    first_names_sample = first_names_cpy.sample(n=num_of_names, replace=True).reset_index(drop=True)

    return first_names_sample

def getRandomSecondName(num_of_names, surnames):
    """
    This function generates a random sample of surnames.

    Parameters:
    num_of_names (int): The number of random surnames to generate.
    surnames (pandas.Series): A Series containing a list of surnames.

    Returns:
    pandas.Series: A Series containing the randomly sampled surnames. The index is reset to a range index.
    The surnames are also converted to title case for consistency.
    """
    surnames_sample = surnames.sample(n=num_of_names, replace=True).reset_index(drop=True)
    surnames_sample = surnames_sample.str.title()
    return surnames_sample

@app.route('/generate_csv_names', methods=["GET"])
def generate_csv_names():
    """
    This function generates a CSV file containing random names based on the given parameters.
    It can generate either first names, surnames, or both, with an optional gender filter.

    Parameters:
    gender (str, optional): The gender of the names to generate. If not provided, names of any gender can be generated.
    num_samples (int): The number of random names to generate.
    first_name (str): A flag indicating whether to generate first names. If set to "True", first names will be included in the output.
    surname (str): A flag indicating whether to generate surnames. If set to "True", surnames will be included in the output.

    Returns:
    flask.Response: A CSV file containing the generated names, with a filename of 'generated_data_normal_minmax.csv'.
    If an error occurs, returns a string describing the error and a 400 status code.
    """
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
    """
    This function generates a CSV file containing random data from a uniform distribution,
    with the minimum and maximum specified by the user.

    Parameters:
    min (float): The minimum value for the generated data. Default is 0.
    max (float): The maximum value for the generated data. Default is 1.
    num_samples (int): The number of samples to generate. Default is 100.

    Returns:
    flask.Response: A CSV file containing the generated data, with a filename of 'generated_data_uniform_minmax.csv'.
    If an error occurs, returns a string describing the error and a 400 status code.
    """
    try:
        minimum = float(request.args.get('min', 0))
        maximum = float(request.args.get('max', 1))
        num_samples = int(request.args.get('num_samples', 100))

        if minimum >= maximum:
            return "Error: 'min' must be less than 'max'", 400

        # Generate uniform distribution data
        data = np.random.uniform(low=minimum, high=maximum, size=num_samples)

        # Humanize the data
        # data = humanize_data(data)

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

@app.route('/generate_csv_w_valprobs', methods=['GET'])
def generate_csv_discrete():
    """
    This function generates a CSV file containing random data from a discrete distribution,
    with the values and their corresponding probabilities specified by the user.

    Parameters:
    values (list): A list of float values representing the possible outcomes of the discrete distribution.
    probabilities (list): A list of float probabilities corresponding to the values. The sum of probabilities must be 1.
    num_samples (int): The number of samples to generate. Default is 100.

    Returns:
    flask.Response: A CSV file containing the generated data, with a filename of 'generated_data_discrete.csv'.
    If an error occurs, returns a string describing the error and a 400 status code.
    """
    try:
        values = request.args.getlist('values', type=float)
        probabilities = request.args.getlist('probabilities', type=float)
        num_samples = int(request.args.get('num_samples', 100))

        if len(values) != len(probabilities):
            return "Error: The number of values must match the number of probabilities", 400

        if abs(sum(probabilities) - 1.0) > 1e-6:
            return "Error: Probabilities must sum to 1", 400

        # Generate discrete distribution data
        data = np.random.choice(values, size=num_samples, p=probabilities)

        # Create a DataFrame
        df = pd.DataFrame({'value': data})

        # Convert DataFrame to CSV
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        # Create the response
        response = make_response(output.getvalue())
        response.headers['Content-Disposition'] = 'attachment; filename=generated_data_discrete.csv'
        response.headers['Content-Type'] = 'text/csv'

        return response
    except Exception as e:
        return str(e), 400

@app.route('/generate_csv_uniform_discrete', methods=['GET'])
def generate_csv_uniform_discrete():
    """
    This function generates a CSV file containing random data from a uniform discrete distribution.

    Parameters:
    min_value (int): The minimum value for the generated data. Default is 0.
    max_value (int): The maximum value for the generated data. Default is 10.
    num_samples (int): The number of samples to generate. Default is 100.

    Returns:
    flask.Response: A CSV file containing the generated data, with a filename of 'generated_data_uniform_discrete.csv'.
    If an error occurs, returns a string describing the error and a 400 status code.
    """
    try:
        min_value = int(request.args.get('min', 0))
        max_value = int(request.args.get('max', 10))
        num_samples = int(request.args.get('num_samples', 100))

        if min_value >= max_value:
            return "Error: 'min' must be less than 'max'", 400

        # Generate uniform discrete data
        data = np.random.randint(low=min_value, high=max_value + 1, size=num_samples)

        # Create a DataFrame
        df = pd.DataFrame({'value': data})

        # Convert DataFrame to CSV
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        # Create the response
        response = make_response(output.getvalue())
        response.headers['Content-Disposition'] = 'attachment; filename=generated_data_uniform_discrete.csv'
        response.headers['Content-Type'] = 'text/csv'

        return response
    except Exception as e:
        return str(e), 400
    
@app.route('/generate_csv_normal_discrete_std', methods=['GET'])
def generate_csv_normal_discrete_std():
    """
    This function generates a CSV file containing random data from a normal discrete distribution.

    Parameters:
    mean (float): The mean of the normal distribution. Default is 0.
    std (float): The standard deviation of the normal distribution. Default is 1.
    num_samples (int): The number of samples to generate. Default is 100.

    Returns:
    flask.Response: A CSV file containing the generated data, with a filename of 'generated_data_normal_discrete_std.csv'.
    If an error occurs, returns a string describing the error and a 400 status code.
    """
    try:
        mean = float(request.args.get('mean', 0))
        std = float(request.args.get('std', 1))
        num_samples = int(request.args.get('num_samples', 100))

        # Generate normal distribution data
        data = np.random.normal(mean, std, num_samples)

        # Round to the nearest integer to get discrete values
        data = np.round(data).astype(int)

        # Create a DataFrame
        df = pd.DataFrame({'value': data})

        # Convert DataFrame to CSV
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        # Create the response
        response = make_response(output.getvalue())
        response.headers['Content-Disposition'] = 'attachment; filename=generated_data_normal_discrete_std.csv'
        response.headers['Content-Type'] = 'text/csv'

        return response
    except Exception as e:
        return str(e), 400

@app.route('/generate_csv_normal_discrete_minmax', methods=['GET'])
def generate_csv_normal_discrete_minmax():
    """
    This function generates a CSV file containing random data from a normal discrete distribution.

    Parameters:
    mean (float): The mean of the normal distribution. Default is 0.
    min (float): The minimum value for the generated data. Default is mean - 1.
    max (float): The maximum value for the generated data. Default is mean + 1.
    num_samples (int): The number of samples to generate. Default is 100.

    Returns:
    flask.Response: A CSV file containing the generated data, with a filename of 'generated_data_normal_discrete_minmax.csv'.
    If an error occurs, returns a string describing the error and a 400 status code.
    """
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

        # Round to the nearest integer to get discrete values
        data = np.round(data).astype(int)

        # Create a DataFrame
        df = pd.DataFrame({'value': data})

        # Convert DataFrame to CSV
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        # Create the response
        response = make_response(output.getvalue())
        response.headers['Content-Disposition'] = 'attachment; filename=generated_data_normal_discrete_minmax.csv'
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


