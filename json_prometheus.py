import json

prom_target = [
    {
        "targets": ["10.2.20.5:8000"],
        "labels": {
            "job": "my_dynamic_device"
        }
    }
]

with open('/home/user/Downloads/prometheus-2.51.2.linux-amd64/targets.json', 'w') as json_file:
    json.dump(prom_target, json_file, indent=4)

json_file.close()
input('[*] Press Enter when getting to the next point...')

with open('/home/user/Downloads/prometheus-2.51.2.linux-amd64/targets.json', 'r') as json_file:
    prom_target = json.load(json_file)

prom_target[0]["targets"] = ["10.2.20.2:8000"]

with open('/home/user/Downloads/prometheus-2.51.2.linux-amd64/targets.json', 'w') as json_file:
    json.dump(prom_target, json_file,indent=4)
json_file.close()