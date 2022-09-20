FROM continuumio/anaconda3:latest
EXPOSE 8000
COPY . /mario
WORKDIR /mario
RUN pip install -r requirements.txt

CMD ["SuperMario.py"]