from flask import Flask, render_template, request, Response

def create_app():
    """an application factory reads config files and objects and create the
    application
    Parameters:
        test_config (dict): The config object that overwrites other setting
        for testing porposes
    Returns:
        Flask: a flask object application"""

    app = Flask(__name__)



    @app.route("/")
    def home():
        return render_template('home.html')


    from .controllers import data_controller
    app.register_blueprint(data_controller, url_prefix='/data')

    return app


if __name__ == '__main__':
    app = create_app()
    app.run()