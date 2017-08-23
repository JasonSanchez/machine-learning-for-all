<?php
header('Content-Type: application/json');

$uploaded = array();

if(!empty($_FILES['file']['name'][0])) {
  foreach($_FILES['file']['name'] as $position => $name) {
    echo $name, '<br>';
  }
}
