FROM python:3.7
MAINTAINER Sohini Roychowdhury <roych@uw.edu>

# Install build utilities
RUN apt-get update && \
    apt-get install -y protobuf-compiler python3-pil python3-lxml python3-pip python3-dev git && \
    apt-get -y upgrade

#Install Object Detection dependencies
RUN python3 -m pip install Flask==1.1.1 WTForms==2.2.1 Flask_WTF==0.14.2 Werkzeug==0.16.0 tensorflow==2.0.0

RUN pip3 install pillow

# Install Object Detection API library
RUN cd /opt && \
    git clone https://github.com/tensorflow/models && \
    cd models/research && \
    protoc object_detection/protos/*.proto --python_out=.
    
RUN cd $HOME && \
    git clone https://github.com/sohiniroych/tensorflow-object-detection-example.git && \
    cp -a tensorflow-object-detection-example/object_detection_app_p2 /opt/ && \
    chmod u+x /opt/object_detection_app_p2/app.py
    
    
RUN python3 /opt/object_detection_app_p2/app.py

    
# expose ports
EXPOSE 8080

#Command
CMD ["python3", "/opt/object_detection_app_p2/app.py", "serve"]
