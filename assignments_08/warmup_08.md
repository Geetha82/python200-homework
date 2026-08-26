# Part 1: Warmup – Cloud Concepts

## Cloud Concepts Question 1

What is the core economic model of cloud computing, and how does it differ from owning your own servers?

Cloud computing works mainly on a pay-as-you-go model, where we pay for the computing resources we use. Instead of buying and maintaining our own servers, we rent resources from a cloud provider, which handles the hardware and maintenance for us.

## Cloud Concepts Question 2

What is the difference between vertical scaling and horizontal scaling? Give a concrete example of when you might choose each.

Vertical scaling means making one machine more powerful by adding resources like RAM, CPU, or a faster GPU. For example, we might upgrade a server with more RAM when an application needs more memory.

Horizontal scaling means adding more machines to handle a larger workload. For example, we  might add more servers to a web app when many more users start accessing it.

- A web app that normally handles 1,000 users per day suddenly needs to handle 100,000 after a viral product launch: **Horizontal scaling**, because more servers can be added to handle the large increase in users.

- A data scientist's model training job is running too slowly, and they want a machine with a faster GPU and more RAM: **Vertical scaling**, because they are making one machine more powerful.

- A data pipeline that processes 10 files per run now needs to process 10,000 files per run, and the work can be split across machines: **Horizontal scaling**, because multiple machines can process the files at the same time.

## Cloud Concepts Question 3
Classify each item in the list below as IaaS, PaaS, SaaS, or BaaS. One sentence of reasoning is enough for each.

### Classifying the Services

* **Gmail — SaaS:** Gmail is software that is ready to use, while Google manages the servers, software, and maintenance.
* **Azure Virtual Machines — IaaS:** Azure provides a virtual machine and infrastructure, while the developer is responsible for managing the operating system and software.
* **AWS S3 (Simple Storage Service) — IaaS:** S3 provides cloud storage infrastructure, and the developer manages the data and how the storage is used.
* **GitHub Codespaces — PaaS:** Codespaces provides a ready-to-use development environment, so the developer can focus on writing and running code instead of managing the underlying infrastructure.
* **Snowflake — SaaS:** Snowflake is a ready-to-use cloud data platform where the provider manages the underlying infrastructure and software.
* **Supabase — BaaS:** Supabase provides backend services such as a database and authentication, so developers do not have to build and manage the backend from scratch.

Describe IaaS, PaaS, and SaaS in your own words:

### IaaS

**IaaS (Infrastructure as a Service)** provides basic computing resources such as virtual machines, storage, and networking. An example is **Azure Virtual Machines**. As a developer, I am responsible for managing things like the operating system, installed software, and my application.

### PaaS

**PaaS (Platform as a Service)** provides a platform where developers can build and run applications without managing the underlying servers. An example is **GitHub Codespaces**. As a developer, I mainly manage my code and application rather than the servers and infrastructure.

### SaaS

**SaaS (Software as a Service)** provides a complete software application that users can access without managing the underlying technology. An example is **Gmail**. As a developer or user, I mainly manage my account, settings, and the data I put into the application while the provider manages the software and infrastructure.

## Cloud Concepts Question 4

What is a managed data platform like Databricks or Snowflake, and how does it differ from using a cloud provider like AWS or GCP directly? What do you gain, and what do you give up?

A managed data platform like Databricks or Snowflake provides tools for storing, processing, and analyzing data without requiring developers to manage all of the underlying infrastructure. Using AWS or GCP directly gives you more control over the servers, storage, and other cloud resources.

The main benefit of a managed data platform is that it is easier to set up and use because the provider handles much of the infrastructure and maintenance. The tradeoff is that you have less control and may pay more for the convenience and specialized services.


## Cloud Concepts Question 5

The lesson names two situations where the cloud is probably not the right choice. What are they?

- The cloud is probably not the right choice when a dataset fits comfortably on one machine and there are no massive computing needs, because local processing can be faster and cheaper. 
- It is also often not the best choice for an initial prototype because using local resources can be simpler than setting up cloud infrastructure.

# Part 2: Warmup – Cloud Landscape

## Cloud Landscape Question 1

Name the three hyperscalers. For each, write one sentence describing its primary strength and the type of organization most likely to use it.

* **AWS (Amazon Web Services):** AWS is known for having a very large selection of cloud services and is commonly used by startups, businesses, and large organizations.

* **Microsoft Azure:** Azure is strong in enterprise and Microsoft-based environments, so it is commonly used by large companies that already use Microsoft products and services.

* **Google Cloud Platform (GCP):** GCP is particularly strong in data, machine learning, and analytics, so it is often used by organizations that work heavily with data and AI.

## Cloud Landscape Question 2

The lesson explains why this course switched from Microsoft Azure to Supabase. It gives three concrete reasons. Summarize each reason in your own words. Then add your own reflection: what does this suggest about how you should evaluate a cloud tool when starting a new project?

* **Access:** Supabase is easier for students to access because they can create an account themselves without waiting for organizational approval or setup.

* **Pedagogical fit:** Supabase uses relational databases with rows and columns, which teaches useful data skills that can be applied to many different jobs.

* **Pipeline coherence:** Supabase makes it easy to store the raw and enriched data in two related tables, which makes the ETL pipeline easier to understand and check.

**Reflection:** When starting a new project, I should choose a cloud tool based on how well it fits the project's needs, how easy it is to use, its cost, and what it will help me accomplish.


## Cloud Landscape Question 3

For each of the four scenarios below, identify which service category from the taxonomy table applies (e.g., "object storage", "managed relational DB", "LLM API", "serverless compute") and name one specific provider or product that offers it.

### Scenario 1: Store 10 TB of image files and retrieve them by filename from any machine
* **Service Category:** object storage
* **Provider/Product:** Amazon S3

### Scenario 2: Run an ML training job on a GPU for four hours, then shut it down
* **Service Category:** serverless compute
* **Provider/Product:** Lambda Labs GPU Cloud

### Scenario 3: Host a web API that automatically scales up when traffic spikes and scales down when it quiets
* **Service Category:** serverless compute
* **Provider/Product:** Google Cloud Run

### Scenario 4: Send structured data to a large language model and get a text response back
* **Service Category:** LLM API
* **Provider/Product:** OpenAI API

## Cloud Landscape Question 4: Multi-Provider Cloud Stack

Describe a simple data project of your own design (one or two sentences is fine) and sketch a plausible stack using services from at least two different providers or products from the taxonomy table. Then answer: is there a benefit to consolidating to one provider, and what would you give up if you did?

### Project Description
I will design a real-time smart security camera application that detects objects in video feeds. The app captures video clips locally, stores them securely in cloud storage, and passes metadata to an AI model to detect anomalies or intruders.

### Plausible Multi-Provider Stack
* **Storage Component:** Amazon S3 (**Provider:** AWS, **Category:** object storage) to reliably store terabytes of raw video files at a low cost.
* **AI & Analytics Component:** OpenAI API (**Provider:** OpenAI, **Category:** LLM API / Vision API) to analyze structured metadata or image frames extracted from the video clips.
* **Backend Component:** Supabase (**Provider:** Supabase, **Category:** managed relational DB / backend-as-a-service) to track user accounts, camera metadata, and alert logs.

### The Benefits and Trade-offs of Consolidation

#### Benefit of Consolidating to One Provider (e.g., all AWS)
* **Simplified Operations:** Consolidating provides unified billing, single-sign-on (SSO) security management, and native integrations that reduce network latency and data egress fees between distinct clouds.

#### What You Give Up by Consolidating
* **Best-of-Breed Tools and Flexibility:** You sacrifice access to superior specialized developer experiences (like Supabase's rapid database tooling) or cutting-edge AI models (like OpenAI's APIs), forcing you to use potentially more complex or less capable single-provider equivalents.

