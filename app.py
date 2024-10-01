from flask import Flask
from routes.upper_teeth import upper_teeth_bp
from routes.lower_teeth import lower_teeth_bp
from routes.detect_caries import caries_bp
from routes.detect_prosthesis import prosthesis_bp

app = Flask(__name__)

# Register blueprints
app.register_blueprint(upper_teeth_bp)
app.register_blueprint(lower_teeth_bp)
app.register_blueprint(caries_bp)
app.register_blueprint(prosthesis_bp)

if __name__ == '__main__':
    app.run(debug=True)
