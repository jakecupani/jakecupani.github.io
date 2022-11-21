// import readingTime from 'reading-time'

type Post = {
  title: string
  file: string
  rawContent: () => string
  tags: () => string
}

export default function getPostData(post: Post) {
  return {
    slug: post.file.split('/').pop().split('.').shift(),
  }
}
