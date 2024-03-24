import 'package:flutter_state_notifier/flutter_state_notifier.dart';
import 'package:instagram_clone/exceptions/custom_exception.dart';
import 'package:instagram_clone/models/comment_model.dart';
import 'package:instagram_clone/providers/comment/comment_state.dart';
import 'package:instagram_clone/repositories/comment_repository.dart';

class CommentProvider extends StateNotifier<CommentState> with LocatorMixin {
  CommentProvider() : super(CommentState.init());

  Future<void> getCommentList({
    required String feedId,
  }) async {
    state = state.copyWith(commentStatus: CommentStatus.fetching);

    try {
      List<CommentModel> commentList =
          await read<CommentRepository>().getCommentList(feedId: feedId);

      state = state.copyWith(
        commentStatus: CommentStatus.success,
        commentList: commentList,
      );
    } on CustomException catch (_) {
      state = state.copyWith(commentStatus: CommentStatus.error);
      rethrow;
    }
  }

  Future<void> uploadComment({
    required String feedId,
    required String uid,
    required String comment,
  }) async {
    state = state.copyWith(commentStatus: CommentStatus.submitting);

    try {
      CommentModel commentModel = await read<CommentRepository>().uploadComment(
        feedId: feedId,
        uid: uid,
        comment: comment,
      );

      state = state.copyWith(
        commentStatus: CommentStatus.success,
        commentList: [commentModel, ...state.commentList],
      );
    } on CustomException catch (_) {
      state = state.copyWith(commentStatus: CommentStatus.error);
      rethrow;
    }
  }
}
