$(window).resize(function() {
    var zoomLevel = window.devicePixelRatio;
    if(zoomLevel < 0.67 && zoomLevel > 0.3){
     $("functions_container").css({ 'height': '3px' });
    }else{
      $("functions_container").css({ 'height': '1px' });
    }
  });