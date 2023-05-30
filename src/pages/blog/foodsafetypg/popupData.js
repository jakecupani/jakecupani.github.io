function popupData(estid) {
    console.log(estid);
    let restaurant;
    for (var i = 0; i < raw_data.data.length; i++) {
        if (raw_data.data[i].establishment_id == estid) {
            console.log(raw_data.data[i]);
            restaurant = raw_data.data[i];
        }
    }
    document.getElementById("name").innerHTML = restaurant.name;
    document.getElementById("address").innerHTML = restaurant.address_line_1;
    // document.getElementById("cleanliness").innerHTML = restaurant.
    document.getElementById("inspection_results").innerHTML = restaurant.inspection_results;
    let api_lookup = restaurant.name + restaurant.address_line_1.split(" ").join("+") + restaurant.city + "+" + restaurant.state + "+" + restaurant.zip;
    console.log(api_lookup);
    let request = "<a target='_blank' id='gmap-request' href='" + "https://www.google.com/maps/search/?api=1&query=" + api_lookup + "'/>";
    console.log(request);
    document.getElementById("location-link").innerHTML = request;
    document.getElementById("gmap-request").innerHTML = "<img id='gmaps-icon' width=20% src='/static/google-maps-logo.png'>";
    document.getElementById("inspection-button").style.display = "block";

    //https://www.google.com/maps/search/?api=1&query=760+West+Genesee+Street+Syracuse+NY+13204
    //loading in detailed Results
    document.getElementById("detailed-results").innerHTML =
        "<li><strong>" + "Inspection Date: </strong>" + "<span class='inspection_color'>" + restaurant.inspection_date.slice(0, -13) + "</span></li>"
        + "<li><strong>" + "Food from approved source: </strong>" + "<span class='inspection_color'>" + restaurant.food_from_approved_source + "</span></li>"
        + "<li><strong>" + "Adequate hand washing : </strong>" + "<span class='inspection_color'>" + restaurant.adequate_hand_washing + "</span></li>"
        + "<li><strong>" + "Proper hand washing : </strong>" + "<span class='inspection_color'>" + restaurant.proper_hand_washing + "</span></li>"
        + "<li><strong>" + "Cold holding temperature: </strong>" + "<span class='inspection_color'>" + restaurant.cold_holding_temperature + "</span></li>"
        + "<li><strong>" + "Cooking time and temperature: </strong>" + "<span class='inspection_color'>" + restaurant.cooking_time_and_temperature + "</span></li>"
        + "<li><strong>" + "Cooling time and temperature: </strong>" + "<span class='inspection_color'>" + restaurant.cooling_time_and_temperature + "</span></li>"
        + "<li><strong>" + "Hot and cold running water: </strong>" + "<span class='inspection_color'>" + restaurant.hot_and_cold_running_water + "</span></li>"
        + "<li><strong>" + "Hot holding temperature: </strong>" + "<span class='inspection_color'>" + restaurant.hot_holding_temperature + "</span></li>"
        + "<li><strong>" + "Ill workers restricted: </strong>" + "<span class='inspection_color'>" + restaurant.ill_workers_restricted + "</span></li>"
        + "<li><strong>" + "No bare-hand contact: </strong>" + "<span class='inspection_color'>" + restaurant.no_bare_hand_contact + "</span></li>"
        + "<li><strong>" + "Proper sewage disposal: </strong>" + "<span class='inspection_color'>" + restaurant.proper_sewage_disposal + "</span></li>"
        + "<li><strong>" + "Reheating time and temperature: </strong>" + "<span class='inspection_color'>" + restaurant.reheating_time_and_temperature + "</span></li>"
        + "<li><strong>" + "Rodent and insects: </strong>" + "<span class='inspection_color'>" + restaurant.rodent_and_insects + "</span></li>";
    inspection_color = document.getElementsByClassName("inspection_color");
    for (i = 0; i < inspection_color.length; i++) {
        if (inspection_color[i].innerHTML.includes("In Compliance")) {
            inspection_color[i].style.color = "green";
        }
        if (inspection_color[i].innerHTML.includes("Out of Compliance")) {
            inspection_color[i].style.color = "red";
        }
    }

}
