import logging
import logstash


class CustomLogStashLogger:

    def __init__(self, logger_name='demo-py-logger', host='0.0.0.0', udp='5959'):

        self.logger_name = logger_name
        self.host = host
        self.udp_port = udp

    def get(self):
        logging.basicConfig(
            filename="logfile",
            filemode="a",
            format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
            level=logging.INFO,
        )

        self.stderrLogger = logging.StreamHandler()
        logging.getLogger().addHandler(self.stderrLogger)
        self.logger = logging.getLogger(self.logger_name)
        self.logger.addHandler(logstash.LogstashHandler(self.host,
                                                        self.udp_port,
                                                        version=1))
        return self.logger


logger_instance = CustomLogStashLogger(udp=5959, host='172.18.0.4', logger_name='akhil')
logger = logger_instance.get()

count = 0
from time import sleep
for i in range(0, 10):
    count = count + 1

    if count % 2 == 0:
        logger.error('Error Message Code Faield :{} '.format(count))
    else:
        logger.info('python-logstash: test logstash info message:{} '.format(count))

