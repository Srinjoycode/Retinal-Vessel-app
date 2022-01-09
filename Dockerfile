FROM bensuperpc/python3-opencv
WORKDIR /app
COPY . /app
COPY requirements.txt ./requirements.txt
RUN apt-get clean && apt-get update && apt-get install -y locales
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
RUN pip install -r requirements.txt
EXPOSE 8501
CMD python app.py