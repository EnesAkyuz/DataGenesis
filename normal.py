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

@app.route('/generate_csv', methods=['GET'])
def generate_csv():
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
        response.headers['Content-Disposition'] = 'attachment; filename=generated_data.csv'
        response.headers['Content-Type'] = 'text/csv'
        
        return response
    except Exception as e:
        return str(e), 400

if __name__ == '__main__':
    app.run(debug=True)
