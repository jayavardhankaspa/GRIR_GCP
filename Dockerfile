FROM tensorflow/tensorflow
COPY ./app_contents /usr/local/python/
EXPOSE 5000
WORKDIR /usr/local/python/
RUN pip install -r requirements.txt
CMD python model_predict_app.py