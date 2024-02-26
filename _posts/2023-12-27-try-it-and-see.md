---
title: Try it and see!
subtitle: Getting better as a SWE, analysis paralysis and why it's so hard for me to accept advice
layout: default
date: 2024-01-04
keywords: software engineering, metacognition, research, applied research
published: true
---

I asked my friend, who is much further in his SWE career, how to improve as a software engineer.

He started with just, **scale**, that I should build projects at "large-scale". "Large-scale", as I later found out, generally means that your service needs to support high loads, respond really fast, or process extremely large amounts of data (sometimes you need to do more than one of these things). To meet these requirements, you make guarantees about your service through metrics like availability, latency or throughput. I initially thought that this was something you only really encounter in infra teams, but my friend disagreed. It's something that can be observed through any project **at scale**.

After a lot of reflection, I realized that he was right! I even had a similar learning experience: scaling [Haystackfs](https://github.com/dhrumilp15/haystackfs), my file search engine for discord, to 380+ servers and 1M+ files.

When I first built the search engine, I tried spinning up ElasticSearch indices on my server. That worked for a few requests, until it would run out of memory. I thought there was something wrong with my code, but that wasn't it - ElasticSearch recommends at least 4gb of RAM, but my server had 1 😂. I needed to optimize for load, but tuning ElasticSearch indices seemed like a painful proposition. So, I instead opted to use Mongodb, which happily handled searching my metadata. Then, users complained about how the bot was super slow. To optimize for latency, I benchmarked a few solutions and ultimately just wrote an engine that would go through chat history to find relevant files. This was a significant speedup for search on recent files, but much slower for extremely old files. I was going to invest in a more complex solution, but users didn't complain about the speed of search (which told me that users generally just need to search recent files).

Around the same time, I read through Discord's engineering blogs on [search](https://discord.com/blog/how-discord-indexes-billions-of-messages) and [storage](https://discord.com/blog/how-discord-stores-trillions-of-messages). They drove decisions about maintaining elastic search indices for search using discussions around latency metrics, storage architecture, etc. Their approach seemed disciplined, but I had no idea how to emulate it on a smaller budget. Still, it was a good example of the kind of engineering philosophy required to build at "large-scale". Reflecting on the experience showed me a lot of what I'd learned about software engineering in trying to scale my bot. I learned to approach hardware limitations, research others' approaches and combine them with my own ideas to create a solution that fit my constraints.

---

When I asked my friend for this advice, I was halfway through my SWE internship at Snowflake on the applied machine learning team. My project was focused around building the first MVP of a clustering function for Snowflake's [Cortex ML functions](https://docs.snowflake.com/en/guides-overview-ml-powered-functions). In the first half of my internship, I built a wrapper around [scikit-learn's k-Means clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html). In the second half, I focused on improving the latency and accuracy of my system. I also worked on an efficient algorithm to [determine the number of clusters in a dataset](https://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set), which is what I want to focus on for this post.

## Trying to estimate the number of clusters in a dataset

When I began my work on methods to estimate the number of clusters in a dataset, I formulated a simple plan based on past research experience:

1. Conduct a literature review to evaluate existing methods for determining the number of clusters in a dataset. Try to answer some of these questions:
    - what is *necessary* to accomplish the task we have in mind?
    - what is the state of the art (SOTA) method? Is the SOTA method tractable given our load and resources?
    - what is the latency and efficacy of a particular method?
2. Choose and implement the best method. 
    - Implement several approaches and iterate until you find an approach that works well
    - Notice that this will typically bring you back to step 1, but there should be fewer things for you to research with every iteration

*I'm always looking to learn more about effective research. If you have any suggestions, please share them (dhrumilp15[at]gmail[dot]com)!*

Unfortunately, in step 1, I would read papers and articles about one particular technique, compare it against the papers and articles explaining another technique, and learn that each technique had its own strengths and drawbacks. I was hoping to find the **best** technique, some sort of holy grail that would always find the right number of clusters but was still tractable for massive customer datasets *(spoiler: this doesn't exist, at least according to my research)*. Even worse, I didn't want to start implementing anything until I'd decided on the algorithm and approach I wanted to take. So, I would compare minute details of algorithms, like whether generating reference distributions from the uniform distribution is better than randomly shuffling features of the data when using the [Gap Statistic](https://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set#The_gap_statistics). When it started to become clear that no one metric was perfect, I thought of combining multiple metrics. The problem was, however, that I **hadn't written any code**. I was debating methods based on notions of the datasets, but didn't have any hard numbers to back it up. This sunk a catastrophic amount of time, because I couldn't commit to any one approach. I had fallen prey to **analysis paralysis**.

In previous internships, my teammates typically shook me out of analysis paralysis by performing the tough task of choosing an approach for me. I'd present all of my analysis, and let them choose the **best** approach. I trusted their judgement, and they generally commented that the approach was more than enough for now and that we should start implementing 😊.

In my time at Snowflake, I was shaken out of analysis paralysis by my performance evaluation, which commented that I needed to take a more active stance in my project and improve my productivity 💀. When I heard the feedback, I was a little bit surprised, but not because I thought that the feedback was wrong. I realized that the feedback could be traced back to a somewhat simple cause: **I just wasn't communicating enough**. I didn't share what I was doing, my progress or what I envisioned for the future of the project. 

I'd been following a working style from a previous internship, where some of the engineering advice I got was counter-productive, so I learned to show finished tasks rather than in-progress work. My performance evaluation at Snowflake showed me that I'd learned categorically wrong lessons - always share your work and agree on a plan to resolve engineering problems with new advice. This doesn't mean that you must follow others' plans over your own, but you should communicate about what you believe to be the best path forward so that you can discover better paths or keep everyone on the same page about why you're doing whatever it is you're doing.

I'd also sunk a lot of time floundering with different approaches for estimating the number of clusters in a dataset. I realized that, like previous internships, if I shared my in-progress analysis with my mentor, we'd see results much faster.

Spurred by my performance evaluation, I rapidly implemented multiple approaches and tested their efficacy on sample datasets. After seeing disappointing performance from multiple algorithms (including the SOTA algorithm), **it finally dawned on me that a perfect approach doesn't exist**. I was genuinely crushed btw, I felt like someone had played a cruel joke on me and recorded my downfall for the world to see.

But, I'd made far more progress understanding the problem, which showed me that research and implementation need to go hand-in-hand. Anytime I compared two different approaches, in the lack of any other easily accessible information, the easiest and often fastest way to decide between them is to just implement both and see the results. In other words, to just

<figure style="text-align: center">
<img src="https://tryitands.ee/tias_thumb.jpg" width="100%" height=auto>
<figcaption>Try it and see! <a href="https://tryitands.ee" target="_blank">Original source</a></figcaption>
</figure>

Implementing multiple approaches also showed me that my internal debates about minute algorithmic details were useless from both a research and practical standpoint. From a research standpoint, I was chasing options that were the least likely to produce sizeable benefits. Through implementation, you can focus your efforts to get the biggest gains upfront, and then later explore overhauls.

Practically, I can guess at the number of clusters forever, but customers will undoubtedly test multiple values. They would also much rather have new features built on top of the clustering function instead of a marginally more accurate estimate for the number of clusters. 


# Does "Try it and see" beat analysis paralysis?

**No.** The truth is, implementation can get you out of analysis paralysis (e.g. lit. review hell), but blindly using *try it and see* will just have you further digging in a bottomless rabbit hole. You have to construct limits around *try it and see*, because you ultimately need to break free from endless exploration and commit to exploiting one particular approach.

I found a surprisingly simple way to combat analysis paralysis when I talked to the caricature artist who had come in for the office party. He drew caricatures in 10 minutes, and he explained that the time constraints taught him how to be confident, develop his own style and not think too hard about small details. His words reminded me of all of the doubt I'd been carrying in my work. His timeout meant that he couldn't afford to agonize over every brush stroke, just like how I couldn't afford to agonize over every particular algorithm. 

Timeouts are also really common in the realm of games, where you also encounter analysis paralysis:
- you could analyze a chess board forever, but a blitz game requires that you do it around 30-40 times in 5 minutes or less. You can't afford to spend all your time analyzing one position, which means that you need to make decisions and adapt.
- other Avalon players will jump on you from a great height if you take forever to choose a team for the mission.

**Try it and see** was my method of escaping the lit. review phase of analysis paralysis, but I fell into a similar cycle while experimenting with different metrics: I tried tweaking several different clustering quality metrics (Davies-Bouldin score, Calinski-Harabasz score, etc.) to see if any of them would produce better results. Instituting a timeout forced me to just choose what seemed to work the best, and then move on to implement the rest of the project. If a better method appeared later, we would just replace the component.

# Have I heard this before?

*Try it and see* is not a novel or unique idea. In writing this post, I thought about the advice I've heard before, and how it encapsulates very similar ideas:

- "Don't wait to gain the skills or knowledge you need to achieve what you want, they will come during the process"
- "The quality of founders is often determined not by the correctness of their decisions, but by how fast they can make them"
- "The best research students are those who can convert an idea to implementation really fast"

Ultimately, these are effectively the same idea, and I've heard them countless times. I then wonder: why has it taken me so long to internalize these ideas even a little bit? It makes me wonder: is advice not advice until you can say the same thing yourself?

I've heard *"do what you love"* too many times to count, but I don't think I truly understand it - how do you balance *"do what you love"* with familial obligations, or a failing economy? It seems like advice doesn't actually become advice until I learn how to apply it, which tends to be when I can finally understand its message and say the same thing to someone else. So, *try it and see*.
