import 'package:flutter/material.dart';
import 'package:instagram_clone/exceptions/custom_exception.dart';

void errorDialogWidget(BuildContext context, CustomException e) {
  showDialog(
    context: context,
    barrierDismissible: false,
    builder: (context) {
      return AlertDialog(
        // 에러 코드
        title: Text(e.code),
        // 에러 내용
        content: Text(e.message),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: Text('확인'),
          ),
        ],
      );
    },
  );
}
