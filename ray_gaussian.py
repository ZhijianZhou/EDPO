
import ray
import logging
import threading
from flask import Flask, request, jsonify
import socket
import time
import subprocess


app = Flask(__name__)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



@app.route('/start_do', methods=['POST'])
def start_do():

    data = request.get_json()
    batch_size = data.get('batch_size', '')

    logger.info(f"The batch size is {batch_size}.")

    
    start_time = time.time()


    refs = [do_gaussian.options(num_cpus=8).remote(task_id) for task_id in range(0,batch_size)]


    while len(refs) != 0:

        readyRefs, unreadyRefs = ray.wait(refs, timeout=60, num_returns=1)

    
        if len(readyRefs) == 0:
            logger.warning(f"Timeout reached. Waiting for results... Remaining tasks: {len(unreadyRefs)}")
        else:

            result = ray.get(readyRefs[0])
            elapsed_time = time.time() - start_time

    
            logger.info(f"Task completed. Result: {result}, Time Elapsed: {elapsed_time:.2f} seconds, Remaining tasks: {len(unreadyRefs)}")

     
            refs = unreadyRefs
        if (time.time()-start_time) > 360 :
            ray.cancel(unreadyRefs)
            break

        time.sleep(0.01)
 
    return jsonify({"message": "Started processing"})



def start_flask_server():
    logger.info("Starting Flask server...")

    host_ip = socket.gethostbyname(socket.gethostname())
    port = 8098


    logger.info(f"Flask server is starting at http://{host_ip}:{port}...")


    curl_command = f'curl -X POST http://{host_ip}:{port}/start_do \\ ' \
                   f'-H "Content-Type: application/json" \\ ' \
                   f'-d \'{{"batch_size": "12"}}\''
    logger.info(f"Generated curl command: {curl_command}")


    app.run(host='0.0.0.0', port=port, threaded=True)


    logger.info(f"Flask server started successfully at http://{host_ip}:{port}.")


@ray.remote
def do_gaussian(idx):
    gif_file = f"/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhouzhejian-240108120128/Mol/e3_diffusion_for_molecules/Temp/{idx}.gjf"
    try:
        subprocess.run(["g16", gif_file], check=True, text=True, capture_output=True)
        return f"The task {idx} process complete."
    except subprocess.CalledProcessError as e:
        with open("/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhouzhejian-240108120128/Mol/e3_diffusion_for_molecules/Temp/{idx}.", "w") as log:
            log.write(f"Error running task {idx}:\n")
        return f"The task {idx} process didn't complete."



if __name__ == "__main__":

    ray.init(address="auto")


    start_flask_server()

    while True:
        time.sleep(60)
