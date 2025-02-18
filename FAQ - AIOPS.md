# FAQ - AIOPS
### http://iops.ai/competition_detail/?competition_id=15&flag=1
##### The link is in one of the support files.


1. If a team has already registered, what is the latest time to make changes and how can I do it?

A: For teams that have already registered, if you want to make changes to your team, the latest you can do is before the end of the preliminaries (mid-April 2020); you can make changes by emailing 重新发送一封报名邮件到netman_iops@126.com with specific information about the changes.

2. When is the deadline for registration?

A: You can register until the end of the preliminaries (mid-April 2020).

3. Will the same data type and format be used for the entire competition phase for the data released in the preliminaries phase?

A: The data type and format will be used in the preliminaries phase. The data may be imperfect, so we hope we can work together to correct it. There may be changes in the final phase, and data types and field attributes, etc. may be changed.

4. Is there any fault injection inside the dataset released in the preliminaries stage?

A: There is no fault injection in the first batch of data in the preliminaries, this data is just to help people get familiar with the data and data format; the subsequent evaluation phase (second batch of data) will release the data after fault injection.

5. How to submit test results in the preliminaries?

A: On the contest page. The original "I want to register" button becomes "submit results" after you click it during registration (only those who have passed the real name authentication have the right to submit). When the test data will be released, we will notify everyone in the group, and then the test results will be submitted here.

**6. At present, the first batch of data does not have positive sample data such as the moment of failure injection and the result of root cause of failure.**

A: There is no fault in the first batch of data, the second batch will be available.

7. Will there be failures that do not occur by active injection?

A: No.

8. What does the negative number of storage free capacity represent?

A: Storage will have a negative number, which means there is not enough space.

9. Is the competition data released now, why is the data downloaded from the official website only 80KB?

A: The sample data is currently downloaded from the link of the web site, the download address is in the folder of the official website download.

**10. Where is the full set of injected failures, moments and durations?**

A: This information will be included in our second batch of release data.

11. Is app1 or app2 related to dcos-container?

A: This mapping will be provided in the second release data.

**12. Inside the call chain, the start time of the parent node is later than the start time of the child node, is this a normal situation?**

A: This problem may be because the clocks between the nodes are not synchronized, the service duration has been recorded in the corresponding field, and can be calculated without StartTime.

13. Why is the same file indicator and server cmdb_id the same, but itemid is different?

A: ConsumerCount is the number of queue consumers, is a class of indicators, the two itemid because it corresponds to a different message queue entity, because there is no queuename in the data, so there will be ambiguity.

14. What is used to draw the call chain diagram?

 A: You can use Graphviz to draw dot diagram, or use Jaeger to draw the trace call path.

15. What is the machine system and version information provided for the preliminary rounds?

A: Cloud resources will not be provided in the preliminary round. The challenge will provide cloud resources to the finalists, and the machine system and version will be Centos 7.2 for the time being.

**16. Why are there two data at the same point in time?**

A: This will happen in the actual data of esb. There is a delay in request recording before the log, when the business volume of the previous time period is counted at regular intervals, if some requests have not been recorded in the log yet, an extra data will be generated at the next count. It is necessary to trouble the player to integrate multiple data of esb at the same moment by himself to calculate the call volume, success volume, average delay, etc.

17. The data file of April 11 does not provide activemq data, won't there be activemq data in the future?

A: The activemq data in the preliminaries has been removed.

18. When is the end of the preliminary round currently scheduled?

A: The end of the preliminary round is expected to be within one month after the official start of the evaluation phase. It is currently planned that the preliminaries will end in late May.

**19. Is there any difference between the data files of the second batch and the first batch? Does it need to be re-downloaded?**

A: The data needs to be re-downloaded, the data contains different fields and the collection time is also different.

**20. There is no application-level call relationship for the second batch of data. Is the application-level call relationship the same as the first batch?**

A: The call chain topology in the second batch data is not the same as the one provided in the first batch.

21. The fault content in the second batch data is April 11, why the data of esb indicators are all on April 10?

A: Please check if the time zone used to convert the timestamp to a string is East 8.

22. If the docker metrics exception has been located, is it the final cause of the exception, or do I need to further locate the metrics exception of the host OS where the docker is located?

A: We can locate the docker indicator exception.

**23. I don't see redis information in the call chain, do I still need to analyze redis metrics for root cause analysis?**

A: redis does not appear in the call chain, you can default osb and redis have a connection.

24. is the failure case in the failure content file all the failure content of the second batch of data, or some examples of failure?

A: The fault file is all the faults.

25. How do I understand the new bomc_id added to the second batch of data? The original said cmdb_id + name can uniquely determine itemid but now it seems like many do not meet the condition.

A: In platform metrics, the (name, bomc_id) binary determines one kind of metrics, and the (name, bomc_id, itemid) ternary determines one kpi curve. The same metric for the same network element (cmdb_id) may correspond to more than one itemid, for example, different tables of the same database.

26. The fault content provided so far feels like the root cause analysis of the fault already, does the player use this fault content to verify the effectiveness of the algorithm? If the root cause detected by the algorithm is exactly the same as here, does it mean the algorithm works well?

A: This time, the data knowledge will help you understand the question and verify the algorithm, and will not evaluate your algorithm. From the next batch of faulty data, the root cause will no longer be provided, and we need to evaluate everyone's detection results.

27. The current data are from 0:00 to 6:00 every day. Will the data for the preliminary and final rounds be available throughout the day, or just for a period of time every day as well?

A: The subsequent specific time range will be adjusted according to the needs of the competition.

28. will there be redis exceptions injected in the data of subsequent competitions?

A: The subsequent failure types are not limited to the currently published scenarios, but we will make sure that each scenario has the final location results.

29. Is the time in the second batch of data the start time or the end time of the failure?

A: The time in the second batch of data is the failure start time.

30. For oracle database, is there a table of mapping relationship between database and host? For example, does db_001 correspond to host os_001?

A: The list of application deployment architectures provided does not include database hosts.

31. Is the database deployed on the virtual machine? There are 22 virtual machines (os_001 to os_022), 6 call chain related ones (os_017 to os_022), 3 redis related ones (os_003 to os_005), and 13 virtual machines for what purpose?

A: The other VMs given in the application deployment architecture list are used for caching, message queues, etc., components that do not appear in the call chain but affect the business. The database is indeed not deployed in these VMs.

32. Excluding the case that bomc_id is empty, is there a one-to-one correspondence between bomc_id and name?

A: bomc_id and name are not one-to-one correspondence.

33. Will there be a second batch of fault data in the preliminaries stage? (The root cause is not given)

A: Yes, there is, the preliminaries are required to be ranked.

**34. csf002~csf005 in "1 application deployment architecture list.xlsx table" belongs to container_002, but csf002~csf005 in "trace_csf.csv" corresponds to cmdb_id is docker_001~docker_004, and docker_001~docker_004 belongs to container_001 (container) in the "1 Application Deployment Architecture List.xlsx" table, that is, the csf type microservices are from two types The data analysis portal matches to different container and docker, how should I use which one?**

A: cmdb_id in the call chain data is the network element that records the data. callType is OSB, RemoteProcess, FlyRemote are recorded by the executor, serviceName corresponds to cmdb_id; callType is CSF, LOCAL are recorded by the initiator The call chain data of callType is CSF and LOCAL is recorded by the initiator, and serviceName is the name of the downstream service; the call chain data of callType is JDBC is also recorded by the initiator, and dsName is the name of the called database.

35. docker's metrics do not have any metrics related to NIC latency. how can I tell that the docker has a NIC latency failure?

A: There is no call chain, which means that there is no service passing through during the whole test period. So you can not test it, and there will be no fault released on it. For the network type failure, if it is judged that the docker is abnormal, there is no need to continue to locate on the indicators; if it is judged to be the host, it is necessary to further give the host-related network indicators. Troubleshooting to the docker problem should be done from the call chain first. At the docker level, if no other docker-related performance metrics such as CPU abnormalities are found, the docker that failed can be output directly. We will determine whether the result is correct this time based on the actual root cause of the failure, if it is a network failure, the result is correct; if not, the result is wrong .

36. The previous platform indicators do not have bomcid , the data format is different, is it not possible to use as fault-free data? Can we only use the six hours data with failure released later?

A: Yes, we will release normal trouble-free data again afterwards.

37. Is the test data after the second batch of data given only the name of the fault (e.g. CPU fault, network fault) and the time?

A: No, the fault name will not be given in the pre-test, and neither will be given in the re-test.

**38. cmdb_id in trace_jdbc is docker_00x, dsName: db_00x, does it mean db_00x is in docker_00x?**

A: It means that docker_00x accesses db_00x. Similarly, in CSF, Local, it means that yes, the cmdb_id object accesses the serviceName of the downstream service.

39. Is the full set of error types is: container CPU utilization type failure, container memory utilization type failure, database type failure, host network type failure, container network type failure these five types of errors, is there no redis error?

A: The preliminaries are these types, you can not deal with redis.

40. There are 11 faults released on May 5, will the evaluation data after that also be of this size? Or will it gradually increase?

A: The number of faults will remain at this size, and will not gradually increase.

41. Can I submit 3 times per day or 3 times in total in each fault release cycle? Is the scoring method for multiple submissions the highest scoring result?

A: You can submit 3 locate results per day; the locate results submitted multiple times will be scored the highest.

42. Will the results of each submission be given back to the contestant in time for the evaluation of each fault release cycle?

A: Yes.

43. Does the full set of faults change with each released dataset?

A: The full set of faults will not change with each released dataset. If there are any changes, the teams will be informed in advance.

44. Are NIC latency and NIC packet loss among the fault types all network faults? If it is a network fault of the host node need to locate to the host specific network indicators?

A: Yes; the network failure of os level needs to be located to specific indicators.

45. How is the preliminaries ranked?

A: The ranking is based on the highest score of the previous submissions.

46. Is fly_remote a call to a service external to the system? If so, is it possible that this call is taking time because of an external system problem? If not, on which node is fly_remote_001 deployed?

A: This does not need to be located, so there is no node listed for deployment.

47. What should I do if it keeps showing uploading when I submit a review?

A: 1. Make sure the submitted results can be run locally with the evaluation script. 2. Check the network environment to ensure that it is stable and that the uploaded results will not be intercepted in the middle of transmission (such as company firewall, etc.). 3. If it still shows that it is being uploaded after ensuring point 1 and 2 please try to submit again.
