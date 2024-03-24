import 'dart:async';

import 'package:flutter/material.dart';

class Debounce {
  final int milliseconds;
  Timer? _timer;

  Debounce({
    required this.milliseconds,
  });

  void run(VoidCallback voidCallback) {
    if (_timer != null) {
      _timer!.cancel();
    }

    // milliseconds 동안 대기 후, voidCallback 실행
    _timer = Timer(Duration(milliseconds: milliseconds), voidCallback);
  }
}
