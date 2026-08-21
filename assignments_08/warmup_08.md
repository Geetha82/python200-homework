# Part 1: Warmup – Cloud Concepts

## Cloud Concepts Question 1

What is the core economic model of cloud computing, and how does it differ from owning your own servers?

Cloud computing works mainly on a pay-as-you-go model, where you pay for the computing resources you use. Instead of buying and maintaining your own servers, you rent resources from a cloud provider, which handles the hardware and maintenance for you.

## Cloud Concepts Question 2

## Cloud Concepts Question 2

What is the difference between vertical scaling and horizontal scaling? Give a concrete example of when you might choose each.

Vertical scaling means making one machine more powerful by adding resources like RAM, CPU, or a faster GPU. For example, you might upgrade a server with more RAM when an application needs more memory.

Horizontal scaling means adding more machines to handle a larger workload. For example, you might add more servers to a web app when many more users start accessing it.

- A web app that normally handles 1,000 users per day suddenly needs to handle 100,000 after a viral product launch: **Horizontal scaling**, because more servers can be added to handle the large increase in users.

- A data scientist's model training job is running too slowly, and they want a machine with a faster GPU and more RAM: **Vertical scaling**, because they are making one machine more powerful.

- A data pipeline that processes 10 files per run now needs to process 10,000 files per run, and the work can be split across machines: **Horizontal scaling**, because multiple machines can process the files at the same time.