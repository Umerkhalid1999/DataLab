"""
Test if dashboard template has Clean Dataset option
"""
from flask import Flask, render_template

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Mock data
datasets = [{
    'id': 1,
    'name': 'test.csv',
    'quality_score': 75,
    'rows': 100,
    'columns': 5,
    'created_at': '2024-01-01',
    'file_type': 'csv',
    'quality_components': {}
}]

@app.route('/test')
def test():
    html = render_template('dashboard.html', 
                          username='Test User',
                          datasets=datasets)
    
    # Check if clean-dataset is in rendered HTML
    if 'clean-dataset' in html:
        return f"""
        <h1 style="color: green;">✓ SUCCESS!</h1>
        <p>The 'clean-dataset' class IS in the rendered HTML!</p>
        <hr>
        <h3>Rendered HTML snippet:</h3>
        <pre>{html[html.find('clean-dataset')-200:html.find('clean-dataset')+200]}</pre>
        <hr>
        <p><a href="/">Go to actual dashboard</a></p>
        """
    else:
        return f"""
        <h1 style="color: red;">✗ FAILED!</h1>
        <p>The 'clean-dataset' class is NOT in the rendered HTML!</p>
        <p>This means there's a template issue.</p>
        """

if __name__ == '__main__':
    print("\n" + "="*60)
    print("TEMPLATE TEST SERVER")
    print("="*60)
    print("\nOpen: http://localhost:5001/test")
    print("\nThis will show if 'clean-dataset' is in the rendered HTML")
    print("="*60 + "\n")
    app.run(port=5001, debug=True)
