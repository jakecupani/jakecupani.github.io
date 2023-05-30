---
title: Food Safety PG Web Application
publishDate: 12 Oct 2021
description: Creating a QR Code Generator with ChatGPT!
tags: ['JavaScript','Web Dev']
---

![FoodSafetyPG](https://raw.githubusercontent.com/jakecupani/foodsafety-pg/master/src/static/pg-logo.png)
# [Github Repo](https://github.com/jakecupani/foodsafety-pg)

> An application created to help users make make informed, healthy decisions regarding where to eat in PG County.

*According to the World Health Organization, 600 million people become ill from contaminated food and around 420 thousand people die from contaminated food every year. Given these statistics, food safety and handling of restaurants is arguably one of the most direct ways in keeping the public healthy.*

*However, the issue stems not only in the lack of cleanliness while preparing food, but the lack of knowledge consumers have while picking places to eat. This lack of access to an easily accessible and intuitive way to determine if the places people go to eat are compliant with food inspection standards is what led to the creation of the "Food Safety PG" application. Its goal is to bring to light the true state of cleanliness of the food establishments to help people make informed decisions about where they go to eat.*

## Our App
Want to know what food establishments are compliant and safe to eat from in PG County? Check out our web application [here](https://www.jakecupani.github.io/blog/foodsafetypg/index.html)!

## Browsers
Our application was created with all browsers in mind.

All modern browsers will be able to host our web-based application, including Chrome, Firefox, & Safari. 

With the growing use of hand-held mobile devices, we created our application with this in mind, paying special attention to the resolution and interactivity of its display. With that being said, our application seemlessly views on either desktop computers or hand-held devices, in order to make the application easy to use wherever it is being viewed. 

In addition, the OS your device is running does not interfere with the ability for the app to preform well because the application has been created with all operating systems in mind. 


## User Manual
Check out our [user manual](https://github.com/jakecupani/foodsafety-pg/blob/master/docs/user.md) to get a more in depth look at our application, as well as how to use it and other useful tips. 

## Developer Manual
Our [developer manual](https://github.com/jakecupani/foodsafety-pg/blob/master/docs/developer_manual.md) has more in-depth information about how to install application dependencies, run the application from the server, run test for the application, and what each endpoint does.  

## Application Dependencies 
The front end of this application is created using simple HTML which provided the foundation and the style/design aspect of the front is done through using CSS and bootstrap library. Leaflet library was used to implement the mapping capabilities for this application which was an important feature. The back-end of this application was made possible through the use to node.js to create a server and connect the API to the server which than supplied data to be displayed in the front-end such as displaying markers on the map and filters. The filters allows the users to manipulate the map to select resturant locations. Steps on how to install application dependencies are included in the developer manual.

## Running Application in the server
Node.js is used to created the server and the prince george's county API supplies the data to the server which is than fetch to the front-end. Depending on the machine you are using, you will need to use PowerShell for windows and termial for Mac to run the server. More in-depth detail on how to run the application from server is found in the [developer manual](https://github.com/jakecupani/foodsafety-pg/blob/master/docs/developer_manual.md) 

## Running Test for the Application
To test our application, we will be using Google Chrome's built in auditing tool. This tool allows us to gauge how well our site is doing in terms of accessibility, performance, best practices, and SEO. Based on our results, we can determine which parts of our application need more work than others. Below we have provided the current output from the auditing tool, which shows our application is doing really well in almost all categories. ![Auditing Results](https://raw.githubusercontent.com/jakecupani/foodsafety-pg/master/src/static/AuditingResults.png)
One thing that was really great about using the Auditing tool was the simulated throttling feature. This allowed us to see how our app would do under low connection circumstances. Thankfully, our app does just fine in these circumstances. We didn't look into much detail on the SEO tips suggested by the tool, but this would definitely be useful should we want to expand this project in the future.

## API for the server application 
The Prince George’s County API was used to get the data to the server for this application. The three endpoints that are implemented in this application includes GET,POST, and PUT endpoints. More detail on what each endpoint does is found on the [developer manual](https://github.com/jakecupani/foodsafety-pg/blob/master/docs/developer_manual.md)

## Bugs and road-map to future developement 
Some of the bugs you might encounter with this application includes loading of the map markers and filters. In the beginning the team had issues with filter. When the user clicked  the filters the map was not populating with markers as expected but it was resolved later. The post request which takes you to the contact page where users can input their contact info but the information is not sent anywhere and for future development that can be worked further to get feedback from users to improve the application. For future development we would like to take this application into a bigger scale, not just PG county but hopefully make it to state level so everyone in Maryland can use this application to make healthy choices when it comes to picking where to eat. We will make changes into the application if we do decide to up the scale of this project by adding different features to the site like adding a search bar so people can easily look up restaurant. We will need more data since we will be covering more areas and solutions to handle massive amount of data on the server and display it to the front of the application without any delay or lag for a smooth user experince. 

### Special Shoutouts
We would like to thank the following citations for allowing our application to be as great as it is. Without their assistance, *Food Safety PG* would not have been as up to date as it is! 



