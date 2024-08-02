from flask import Flask, request, make_response
import pandas as pd
import numpy as np
import io

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

if __name__ == '__main__':
    app.run(debug=True)

# TESTING:  curl -o generated_data_normal_std.csv "http://127.0.0.1:5000/generate_csv_normal_std?mean=10&std=5&num_samples=1000"
# TESTING:  curl -o generated_data_normal_minmax.csv "http://127.0.0.1:5000/generate_csv_normal_minmax?mean=10&min=5&max=15&num_samples=1000"