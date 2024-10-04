document.addEventListener("DOMContentLoaded", function() {
    const pathName = window.location.pathname;
    const pageName = pathName.split("/").pop();

    console.log("Pathname:", pathName);
    console.log("Page name:", pageName);

    if(pageName === "index") {
        // console.log("Highlighting Home");
        document.querySelector(".home").classList.add("activeLink");
    }
    if(pageName === "about_us") {
        // console.log("Highlighting About Us");
        document.querySelector(".about_us").classList.add("activeLink");
    }
    if(pageName === "building-1") {
        // console.log("Highlighting Building 1");
        document.querySelector(".building_1").classList.add("activeLink");
    }
    if(pageName === "building-2") {
        // console.log("Highlighting Building 2");
        document.querySelector(".building_2").classList.add("activeLink");
    }
    if(pageName === "building-3") {
        // console.log("Highlighting Building 3");
        document.querySelector(".building_3").classList.add("activeLink");
    }
    if(pageName === "model-comparison") {
        // console.log("Highlighting Model Comparison");
        document.querySelector(".model_comparison").classList.add("activeLink");
    }
});
