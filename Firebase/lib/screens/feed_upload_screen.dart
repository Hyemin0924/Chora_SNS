import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:instagram_clone/exceptions/custom_exception.dart';
import 'package:instagram_clone/providers/feed/feed_provider.dart';
import 'package:instagram_clone/providers/feed/feed_state.dart';
import 'package:instagram_clone/widgets/error_dialog_widget.dart';
import 'package:provider/provider.dart';

class FeedUploadScreen extends StatefulWidget {
  final VoidCallback onFeedUploaded;

  const FeedUploadScreen({
    super.key,
    required this.onFeedUploaded,
  });

  @override
  State<FeedUploadScreen> createState() => _FeedUploadScreenState();
}

class _FeedUploadScreenState extends State<FeedUploadScreen> {
  final TextEditingController _textEditingController = TextEditingController();
  final List<String> _files = [];

  Future<List<String>> selectImages() async {
    List<XFile> images = await ImagePicker().pickMultiImage(
      maxHeight: 1024,
      maxWidth: 1024,
    );
    return images.map((e) => e.path).toList();
  }

  List<Widget> selectedImageList() {
    final feedStatus = context.watch<FeedState>().feedStatus;

    return _files.map((data) {
      return Padding(
        padding: const EdgeInsets.only(left: 10),
        child: Stack(
          children: [
            ClipRRect(
              child: Image.file(
                File(data),
                fit: BoxFit.cover,
                height: MediaQuery.of(context).size.height * 0.4,
                width: 280,
              ),
              borderRadius: BorderRadius.circular(5),
            ),
            Positioned(
              top: 10,
              right: 10,
              child: InkWell(
                onTap: feedStatus == FeedStatus.submitting
                    ? null
                    : () {
                        setState(() {
                          _files.remove(data);
                        });
                      },
                child: Container(
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.4),
                    borderRadius: BorderRadius.circular(60),
                  ),
                  height: 30,
                  width: 30,
                  child: Icon(
                    color: Colors.black.withOpacity(0.6),
                    size: 30,
                    Icons.highlight_remove_outlined,
                  ),
                ),
              ),
            ),
          ],
        ),
      );
    }).toList();
  }

  @override
  void dispose() {
    _textEditingController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final feedStatus = context.watch<FeedState>().feedStatus;

    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        automaticallyImplyLeading: false,
        actions: [
          TextButton(
            onPressed:
                (_files.length == 0 || feedStatus == FeedStatus.submitting)
                    ? null
                    : () async {
                        try {
                          FocusScope.of(context).unfocus();

                          await context.read<FeedProvider>().uploadFeed(
                                files: _files,
                                desc: _textEditingController.text,
                              );

                          ScaffoldMessenger.of(context).showSnackBar(
                            SnackBar(content: Text('Feed를 등록했습니다.')),
                          );

                          widget.onFeedUploaded();
                        } on CustomException catch (e) {
                          errorDialogWidget(context, e);
                        }
                      },
            child: Text('Feed'),
          ),
        ],
      ),
      body: GestureDetector(
        onTap: () => FocusScope.of(context).unfocus(),
        child: Container(
          padding: const EdgeInsets.symmetric(
            vertical: 15,
            horizontal: 15,
          ),
          alignment: Alignment.topCenter,
          child: SingleChildScrollView(
            child: Column(
              children: [
                LinearProgressIndicator(
                  backgroundColor: Colors.transparent,
                  value: feedStatus == FeedStatus.submitting ? null : 1,
                  color: feedStatus == FeedStatus.submitting
                      ? Colors.red
                      : Colors.transparent,
                ),
                SingleChildScrollView(
                  scrollDirection: Axis.horizontal,
                  child: Row(
                    children: [
                      InkWell(
                        onTap: feedStatus == FeedStatus.submitting
                            ? null
                            : () async {
                                final _images = await selectImages();
                                setState(() {
                                  _files.addAll(_images);
                                });
                              },
                        child: Container(
                          height: 80,
                          width: 80,
                          child: const Icon(Icons.upload),
                          decoration: BoxDecoration(
                            color: Colors.white.withOpacity(0.2),
                            borderRadius: BorderRadius.circular(5),
                          ),
                        ),
                      ),
                      ...selectedImageList(),
                    ],
                  ),
                ),
                if (_files.isNotEmpty)
                  TextFormField(
                    controller: _textEditingController,
                    decoration: const InputDecoration(
                      hintText: '내용을 입력하세요...',
                      border: InputBorder.none,
                    ),
                    maxLines: 5,
                  ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
