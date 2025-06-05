FROM localstack/localstack:latest

RUN mkdir -p /etc/localstack/init/ready.d

COPY localstack-init/01-create-resources.sh /etc/localstack/init/ready.d/
COPY localstack-init/lambda_function.py /etc/localstack/init/ready.d/

RUN chmod +x /etc/localstack/init/ready.d/*.sh