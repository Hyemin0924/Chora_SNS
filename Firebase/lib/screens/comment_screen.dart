import 'package:flutter/material.dart';
import 'package:instagram_clone/exceptions/custom_exception.dart';
import 'package:instagram_clone/models/user_model.dart';
import 'package:instagram_clone/providers/comment/comment_provider.dart';
import 'package:instagram_clone/providers/comment/comment_state.dart';
import 'package:instagram_clone/providers/user/user_state.dart';
import 'package:instagram_clone/widgets/avatar_widget.dart';
import 'package:instagram_clone/widgets/comment_card_widget.dart';
import 'package:instagram_clone/widgets/error_dialog_widget.dart';
import 'package:provider/provider.dart';

class CommentScreen extends StatefulWidget {
  final String feedId;

  const CommentScreen({
    super.key,
    required this.feedId,
  });

  @override
  State<CommentScreen> createState() => _CommentScreenState();
}

class _CommentScreenState extends State<CommentScreen> {
  GlobalKey<FormState> _formKey = GlobalKey<FormState>();
  TextEditingController _textEditingController = TextEditingController();
  late final CommentProvider commentProvider;

  @override
  void initState() {
    super.initState();
    commentProvider = context.read<CommentProvider>();
    _getCommentList();
  }

  void _getCommentList() {
    WidgetsBinding.instance.addPostFrameCallback((_) async {
      try {
        await commentProvider.getCommentList(feedId: widget.feedId);
      } on CustomException catch (e) {
        errorDialogWidget(context, e);
      }
    });
  }

  @override
  void dispose() {
    _textEditingController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    UserModel currentUserModel = context.read<UserState>().userModel;
    CommentState commentState = context.watch<CommentState>();
    bool isEnabled = commentState.commentStatus != CommentStatus.submitting;

    if (commentState.commentStatus == CommentStatus.fetching) {
      return Center(
        child: CircularProgressIndicator(),
      );
    }

    return Scaffold(
      body: SafeArea(
        child: ListView.builder(
          itemCount: commentState.commentList.length,
          itemBuilder: (context, index) {
            return CommentCardWidget(
                commentModel: commentState.commentList[index]);
          },
        ),
      ),
      bottomNavigationBar: Container(
        margin:
            EdgeInsets.only(bottom: MediaQuery.of(context).viewInsets.bottom),
        color: Colors.black54,
        child: Form(
          key: _formKey,
          child: Row(
            children: [
              AvatarWidget(userModel: currentUserModel),
              Expanded(
                child: Padding(
                  padding: const EdgeInsets.only(left: 16, right: 8),
                  child: TextFormField(
                    controller: _textEditingController,
                    enabled: isEnabled,
                    decoration: InputDecoration(
                      hintText: '댓글을 입력해주세요',
                      border: InputBorder.none,
                    ),
                    validator: (value) {
                      if (value == null || value.trim().isEmpty) {
                        return '댓글을 입력해주세요';
                      }
                      return null;
                    },
                  ),
                ),
              ),
              IconButton(
                onPressed: isEnabled ? () async {
                  FocusScope.of(context).unfocus();

                  FormState? form = _formKey.currentState;

                  if (form == null || !form.validate()) {
                    return;
                  }

                  try {
                    // 댓글 등록 로직
                    await context.read<CommentProvider>().uploadComment(
                          feedId: widget.feedId,
                          uid: currentUserModel.uid,
                          comment: _textEditingController.text,
                        );
                  } on CustomException catch (e) {
                    errorDialogWidget(context, e);
                  }

                  _textEditingController.clear();
                } : null,
                icon: Icon(Icons.comment),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
