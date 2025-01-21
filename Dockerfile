FROM python:3.10-bullseye


# 修改时区
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


RUN sed -i "s@http://\(deb\|security\).debian.org@https://mirrors.aliyun.com@g" /etc/apt/sources.list

# 安装依赖库
RUN apt update && \
    apt install tzdata -y

# 更新时区配置
RUN dpkg-reconfigure --frontend noninteractive tzdata


RUN mkdir /app
WORKDIR /app
COPY ./src /app/src
COPY ./pyproject.toml /app/pyproject.toml
COPY ./README.md /app/README.md

# install translation-agent with latest code
RUN python -m pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/ && \
  pip install .

EXPOSE 8000

CMD ["python3", "-m", "translation_agent.web_api", "--port",  "8000"]