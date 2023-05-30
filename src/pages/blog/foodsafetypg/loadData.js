let markers;
let raw_data;

function resetMap() {
    try {
        markers.clearLayers();
    }
    catch {
        console.log('no markers made yet');
    }
}
function loadData(filters) {
    // confirm code is running on click
    fetch('/api')
        .then(res => res.json())
        .then(res => {
            raw_data = res;
            console.log("the raw data", raw_data);
            if (filters) {
                console.log("there are filters");
                newdata = res.data.filter(el => filters.includes(el.category));
            }
            else {
                console.log("there are no filters");
                newdata = res.data;
            }
            console.log(newdata);
            return newdata;

        })
        .then(res => {
            let places = new Array();
            let names = new Array();
            for (i = 0; i < res.length; i++) {
                let obj = res[i]["geocoded_column_1"];
                try {
                    places.push(L.geoJSON(obj).bindPopup("<p>" + res[i].name + "</p><button class = 'btn btn-block btn-primary' onclick='popupData(" + res[i].establishment_id + ")'>Get Results</button>"));
                    names.push(res[i].name + res[i].address_line_1);
                }
                catch {
                    console.log("This did not work");
                    continue;
                }

            }
            resetMap();
            markers = L.layerGroup(places).addTo(mymap);
            return res;
        });
}
