function filterData() {
    console.log('filtering');
    let checkboxes = document.getElementsByName("filter-option");
    let filters = new Array();
    for (i = 0; i < checkboxes.length; i++) {
        if (checkboxes[i].checked) {
            filters.push(checkboxes[i].value);
        }
    }
    loadData(filters);
}
