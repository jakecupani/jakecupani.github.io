let mymap = L.map('mapid').setView([38.9282, -76.9158], 13);

L.tileLayer('https://api.mapbox.com/styles/v1/{id}/tiles/{z}/{x}/{y}?access_token={accessToken}', {
    attribution: 'Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, Imagery © <a href="https://www.mapbox.com/">Mapbox</a>',
    maxZoom: 18,
    id: 'mapbox/streets-v11',
    tileSize: 512,
    zoomOffset: -1,
    accessToken: 'pk.eyJ1IjoieHhoeWRyYXgiLCJhIjoiY2t1b3N3OXpkNGVsNjJxbnpiNTh6M3hqMCJ9.yanVbgvA5-MpTgB9eKS5Og'
}).addTo(mymap);
