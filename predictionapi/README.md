# ML Model - Iris Classification Prediction API using Flask
This code contains one Python REST API using Flask along with CORS enabled for a machine learning model Iris Classification Prediction.


## Quickstart
* Activate the virtual environment: **source app_env/bin/activate**
* Change to mlmodel_iris directory: **cd mlmodel_iris**
* To run the application in linux system: **uwsgi --ini uwsgi_config.ini**
* Deactivate the virtual environment: **deactivate**


## API call using Postman
* POST http://localhost:5080/iris/predict
<br>Request Header: Content-Type : text/plain<br>
<br>Request Body:<br>
{
	"petal_lengths" : ["4.0", "7.0", "6.3"],
	"petal_widths" : ["2.8", "3.2", "2.7"],
	"sepal_lengths" : ["1.0", "4.7", "4.9"],
	"sepal_widths" : ["0.1", "1.4", "1.8"]
}
<br>
<br>Response:
<br>Status Code : 200
<br>Response Body:<br>
{
    "data": [
        "Iris-setosa",
        "Iris-versicolor",
        "Iris-virginica"
    ],
    "testaccuracy": "96.86609688307824"
}


## API Deployment
Next, create the systemd service unit file ending in .service within the /etc/systemd/system directory. Creating a systemd unit file will allow Ubuntu's init system to automatically start uWSGI and serve the Flask application whenever the server boots:

```shell
sudo nano /etc/systemd/system/mlmodel_iris.service
```

As the editor opens the file, write the following:

```shell
[Unit]
Description=uWSGI instance to serve mlmodel_iris
After=network.target

[Service]
User=root <or your username>
Group=www-data
WorkingDirectory=/root/home/deployedmodels/irismodel/mlmodel_iris
Environment="PATH=/root/home/deployedmodels/irismodel/app_env/bin"
ExecStart=/root/home/deployedmodels/irismodel/app_env/bin/uwsgi --ini uwsgi_config.ini

[Install]
WantedBy=multi-user.target
```

Save and close it.
Now start the uWSGI service we created and enable it so that it starts at boot:

```shell
sudo systemctl start mlmodel_iris
sudo systemctl enable mlmodel_iris
```

Check the status:

```shell
sudo systemctl status mlmodel_iris
```


## First time with Nginx
To configure Nginx to proxy requests. Create a new server block configuration file **mlmodels** in Nginx's sites-available directory:

```shell
sudo nano /etc/nginx/sites-available/mlmodels
```

As the editor opens the file, write the following:

```shell
server {
    listen       80;
    server_name  yourserver.com;

    location /iris/predict {
        proxy_pass http://127.0.0.1:5080/iris/predict;
    }
}
```

Save and close it.
Now enable the Nginx server block configuration by linking the file to the sites-enabled directory:

```shell
sudo ln -s /etc/nginx/sites-available/mlmodels /etc/nginx/sites-enabled
```

Compile the nginx config files:

```shell
sudo nginx -t
```

Restart the Nginx process to read the new configuration:

```shell
sudo systemctl restart nginx
```

The REST API is ready to be consumed at:
http://yourserver.com/iris/predict

## Make it secure

To make it https, first, add the Certbot Ubuntu repository:

```shell
sudo add-apt-repository ppa:certbot/certbot
```

Next, install Certbot's Nginx package with apt:

```shell
sudo apt install python-certbot-nginx
```

Then, create certficate for the domain yourserver.com:

```shell
sudo certbot --nginx -d yourserver.com
```

Press 2 to redirect all requests to https.
That's it.

Now the REST API is ready to be consumed at:
https://yourserver.com/iris/predict


## Second time with Nginx
To configure Nginx to proxy requests. Open the **mlmodels** server block configuration file in Nginx's sites-available directory:

```shell
sudo nano /etc/nginx/sites-available/mlmodels
```

As the editor opens the file, write the following:

```shell
server {

    location /project1/predict {
        proxy_pass http://127.0.0.1:5070/project1/predict;
    }

    # Add new location block for new application
    location /iris/predict {
        proxy_pass http://127.0.0.1:5080/iris/predict;
    }

    listen 443 ssl; # managed by Certbot
    ssl_certificate /etc/letsencrypt/live/yourserver.com-0001/fullchain.pem; # managed by Certbot
    ssl_certificate_key /etc/letsencrypt/live/yourserver.com-0001/privkey.pem; # managed by Certbot
    include /etc/letsencrypt/options-ssl-nginx.conf; # managed by Certbot
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem; # managed by Certbot

}server {
    if ($host = yourserver.com) {
        return 301 https://$host$request_uri;
    } # managed by Certbot


    listen       80;
    server_name  yourserver.com;
    return 404; # managed by Certbot


}
```

Save and close it.
Compile the nginx config files:

```shell
sudo nginx -t
```

Restart the Nginx process to read the new configuration:

```shell
sudo systemctl restart nginx
```

As the domain is already secured, no need to do it again.

The REST API is ready to be consumed at:
https://yourserver.com/iris/predict