#ifndef GASPICXX_HPP_
#define GASPICXX_HPP_

#include <memory>
#include <list>
#include <set>

extern "C" {
#include <GASPI.h>
}

//// fowrward declarations
namespace gaspi {

namespace group {

class Group;
class Rank;

}

namespace segment { class Segment; class NotificationManager;}
namespace passive { class Passive; }

namespace singlesided {

class BufferDescription;
class Queue;

}
}


namespace gaspi {

using Rank = gaspi_rank_t;

class Context
{
  private:

    //! The rank of "this" process
    gaspi_rank_t _rank;

    std::unique_ptr<group::Group>       _pGroup;
    std::unique_ptr<singlesided::Queue> _pQueue;

    //! A communicator cannot be copied.
    Context
      (const Context&) = delete;

    Context&
    operator=
      (const Context&) = delete;

  public:

    // default context, all ranks
    Context
      ();

    Context
      (group::Group && group);

    virtual
    ~Context();

    /// Returns the rank of this process in the communicator
    group::Rank
    rank
      () const;

    /// Returns the size of this communicator
    group::Rank
    size
      () const;

    group::Group const &
    group
      () const;

    void
    write
      ( singlesided::BufferDescription sourceBufferDescription
      , singlesided::BufferDescription targetBufferDescription ) const;

    void
    notify
      ( singlesided::BufferDescription targetBufferDescription ) const;

    bool
    checkForBufferNotification
      ( singlesided::BufferDescription targetBufferDescription ) const;

    bool
    waitForBufferNotification
      ( singlesided::BufferDescription targetBufferDescription ) const;

    void
    flush
      () const;

    /// Collective barrier call for all processes in `this` communicator.
    void
    barrier() const;
};

class RuntimeBase {

   public:

     RuntimeBase
       ();

     ~RuntimeBase
       ();
 };

//! This class provides an runtime for GASPI communication.
//! It basically combines a GASPI group and a GASPI segment.
//! \brief   Abstraction of a GASPI segment and group
//! \warning Each GASPI interface must have its own GASPI segment
//!          and GASPI queue that are not used by the user
//!          application in any other way!
//!          Therefore, interfaces do not provide a copy constructor
//!          or an assignment operator.
class Runtime : public RuntimeBase
              , public Context
{
private:

  std::unique_ptr<segment::Segment> _psegment;
  std::unique_ptr<passive::Passive> _ppassive;

  //! A runtime cannot be copied.
  Runtime
    (const Runtime&) = delete;

  Runtime&
  operator=(const Runtime&) = delete;


public:

  using Rank = gaspi_rank_t;

  //! Construct a GASPI interface from a group and a segment.
  //! \note GASPI and the given segment must be initialized on
  //!       all ranks of the given group!
  Runtime
    ();
  //! The destructor
  ~Runtime
    ();

  //! Return the segment
  segment::Segment &
  segment()  {
    return *_psegment;
  }

  passive::Passive &
  passive() {
    return *_ppassive;
  }
};

bool
isRuntimeAvailable();

Runtime &
getRuntime();

namespace group {

class Rank {

public:

  using Type = unsigned short;

  explicit
  Rank
    ( Type rank );

  Type
  get
    () const;

  Rank &
  operator++();

  Rank
  operator++(int);

  Rank &
  operator--();

  Rank
  operator--(int);

  Rank
  operator+(Rank const & other) const;

  Rank
  operator+(int const & other) const;

  Rank
  operator-(Rank const & other) const;

  Rank
  operator-(int const & other) const;

  Rank
  operator%(Rank const & other) const;

  bool
  operator==( Rank const & other ) const;

  bool
  operator!=( Rank const & other ) const;

  bool
  operator<( Rank const & other ) const;

  bool
  operator<=( Rank const & other ) const;

  bool
  operator>( Rank const & other ) const;

  bool
  operator>=( Rank const & other ) const;

private:

  Type _rank;

};

class Group
{
  private:

    //! The group of ranks that constitute the interface
    std::unique_ptr<gaspi_group_t> _pgroup;

  public:

    Group
      ();

    Group
      (Group &&);

    Group
      (std::set<gaspi_rank_t> const &);

    virtual
    ~Group();

    gaspi_group_t const &
    group
      () const;

    Rank
    size
      () const;

//    bool
//    contains
//      () const;

    Rank
    rank() const;
};

gaspi_rank_t
groupToGlobalRank
  ( Group const & group
  , Rank const & rank );

Rank
globalToGroupRank
  ( Group const & group
  , gaspi_rank_t const & rank );

} /* namespace group */

namespace segment {

using SegmentID = int;
using Notification = int;

class MemoryManager {

public:

  MemoryManager
    ( void * const memory_segment_ptr
    , std::size_t  memory_segment_size );

  ~MemoryManager();

  void *
  allocate
    ( std::size_t size );

  void
  deallocate
    ( void * ptr
    , std::size_t size );

  friend std::ostream&
  operator<<
    ( std::ostream& os
    , const MemoryManager& man );

private:

  MemoryManager
    ( MemoryManager const & ) = delete;

  MemoryManager &
  operator=
    ( MemoryManager const & ) = delete;

  struct MemoryBlock
  {
         void   *g_start;
         size_t size;
         bool   free;
  };

  std::list<MemoryBlock> _blocks;
  pthread_mutex_t        _mutex;

};

template <typename T = void>
class Allocator
{
  public:

    template <typename U> friend struct Allocator;

    using value_type = T;
    using pointer = T *;

    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;

    explicit
    Allocator
      ( MemoryManager * const memory )
    : _memory(memory)
    {}

    template <typename U>
    Allocator
      ( Allocator<U> const & rhs )
    : _memory(rhs._memory)
    {}

    pointer
    allocate
      ( std::size_t n )
    {
      return static_cast<pointer>(_memory->allocate(n * sizeof(T)));
    }

    void
    deallocate
      ( pointer p
      , std::size_t n)
    {
        _memory->deallocate(p, n * sizeof(T));
    }

    template <typename U>
    bool
    operator==
      (Allocator<U> const & rhs) const
    {
      return _memory == rhs._memory;
    }

    template <typename U>
    bool
    operator!=
      (Allocator<U> const & rhs) const
    {
      return _memory != rhs._memory;
    }

private:

    MemoryManager * const _memory;

};

class SegmentManager
{
  public:

    using SegmentID = int;
    using Notification = int;

    SegmentManager
      () = delete;

    SegmentManager
      ( SegmentID segmentID );

    ~SegmentManager
      ();

    SegmentID
    id
      () const;

    std::size_t
    size
      () const;

    std::size_t
    pointerToOffset
      ( void const * const ) const;

    Allocator<char>
    allocator
      ();

    Notification
    acquire_notification
      ();

    void
    release_notification
      (Notification const & notification);

private:

    SegmentID                            _segmentID;
    std::unique_ptr<MemoryManager>       _memoryManager;
    std::unique_ptr<NotificationManager> _notifyManager;

};

class SegmentResource
{
  public:

    using SegmentID = SegmentManager::SegmentID;

    SegmentResource
      ( std::size_t segmentSize );

    SegmentResource
      ( SegmentID segmentId
      , std::size_t segmentSize );

    ~SegmentResource
      ();

    SegmentID
    id
      () const;

    void
    remoteRegistration
      ( Rank );

    static SegmentID
    getFreeLocalSegmentId();

private:

    SegmentID _segmentId;

};

class Segment : public SegmentResource
              , public SegmentManager
{
  public:

    using SegmentID = SegmentManager::SegmentID;
    using Notification = SegmentManager::Notification;

    Segment
      () = delete;

    Segment
      ( std::size_t );

    Segment
      ( SegmentID
      , std::size_t );

    SegmentID
    id
      () const
    {
      return SegmentResource::id();
    }

private:

};

} // segment

//! This class implements the RAII idiom for memory
//! allocations on GASPI segments.
//! \brief Managed memory allocation on a GASPI segment
template < class T
         , template <typename> class Allocator = segment::Allocator >
class ScopedAllocation
{
  private:

    Allocator<T>   _allocator;
    T * const      _g_pointer;
    std::size_t    _count;

public:
  //! Allocate <c>count*sizeof(T)</c> bytes on the given interface.
  //! \brief Constructor
  ScopedAllocation
    ( Allocator<T> const & allocator
    , std::size_t count)
  : _allocator(allocator)
  , _g_pointer
      (_allocator.allocate(count))
  , _count(count)
  { }

  ~ScopedAllocation
    ()
  {
    _allocator.deallocate(_g_pointer,_count);
  }
  //! \brief   Get a pointer to the allocated memory.
  //! \warning Do not take ownership of the pointer and make sure
  //!          it is not used outside of the current scope!
  T *
  pointer() const
    { return _g_pointer; }

  //! Access the allocated memory.
  T&
  operator[]
    (int index) const
  { return _g_pointer[index]; }

};

namespace singlesided {

class BufferDescription;
class MemoryAllocation;
class NotifyAllocation;

class Buffer {

  public:

    Buffer
      ( segment::Segment & segment
      , std::size_t size );

    Buffer
      ( void * const ptr
      , segment::Segment & segment
      , std::size_t size );

    Buffer
      ( segment::Segment & segment
      , std::size_t size
      , segment
          ::Notification notification );

    Buffer
      ( void * const ptr
      , segment::Segment & segment
      , std::size_t size
      , segment
          ::Notification notification );

    ~Buffer
      ();

    BufferDescription
    description
      () const;

    void *
    address
      () const;

    // Checks for notification
    // return true if thread got notification (only a single thread gets the
    //        notification)
    bool
    checkForNotification
      ();

    // Waits for notification
    // return true if thread got notification (only a single thread gets the
    //        notification)
    bool
    waitForNotification
      ();

  protected:

    std::unique_ptr<MemoryAllocation>  _allocMemory;
    std::unique_ptr<NotifyAllocation>  _allocNotify;

    void * const        _pointer;
    std::size_t         _size;
    segment
      ::Notification    _notification;
    segment::Segment &  _segment;

};


class Endpoint : public Buffer {

  public:

    using Tag = int;

    class ConnectHandle {

      public :

        ConnectHandle
          ( Endpoint & commBuffer
          , std::unique_ptr<Buffer> pSendBuffer
          , std::unique_ptr<Buffer> pRecvBuffer );

        void
        waitForCompletion();

      private:

        Endpoint & _commBuffer;
        std::unique_ptr<Buffer> _pSendBuffer;
        std::unique_ptr<Buffer> _pRecvBuffer;

    };

    Endpoint
      ( segment::Segment & segment
      , std::size_t size );

    Endpoint
      ( void * const ptr
      , segment::Segment & segment
      , std::size_t size );

    Endpoint
      ( segment::Segment & segment
      , std::size_t size
      , segment
          ::Notification notification );

    Endpoint
      ( void * const ptr
      , segment::Segment & segment
      , std::size_t size
      , segment
          ::Notification notification );

    ~Endpoint();

    void
    setRemotePartner
      ( BufferDescription const & partnerDescription );

    // bilateral function
    // needs to be invoked by the correspondent
    // WriteTargetBuffer having the same size
    ConnectHandle
    connectToRemotePartner
      ( Context & context
      , group::Rank & rank
      , Tag & tag );

    bool
    isConnected
      () const;

  protected:

    BufferDescription &
    localBufferDesc();

    BufferDescription const &
    localBufferDesc() const;

    BufferDescription &
    otherBufferDesc();

    BufferDescription const &
    otherBufferDesc() const;

  private:

    std::unique_ptr<BufferDescription>   _pLocalBufferDesc;
    std::unique_ptr<BufferDescription>   _pOtherBufferDesc;
    bool                                 _isConnected;

};

namespace write {

class SourceBuffer : public Endpoint {

  public:

    using Tag = int;

    SourceBuffer
      ( segment::Segment & segment
      , std::size_t size );

    SourceBuffer
      ( void * const ptr
      , segment::Segment & segment
      , std::size_t size );

    SourceBuffer
      ( segment::Segment & segment
      , std::size_t size
      , segment::Segment
          ::Notification notification );

    SourceBuffer
      ( void * const ptr
      , segment::Segment & segment
      , std::size_t size
      , segment::Segment
          ::Notification notification );

    ~SourceBuffer
      ();

    // bilateral function
    // needs to be invoked by the correspondent
    // WriteTargetBuffer having the same size
    Endpoint::ConnectHandle
    connectToRemoteTarget
      ( Context & context
      , group::Rank & rank
      , Tag & tag );

    void
    initTransfer
      ( Context & context );

    bool
    checkForTransferAck
      ( );

    bool
    waitForTransferAck
      ( );

};

class TargetBuffer : public Endpoint {

  public:

    using Tag = int;

    // Allocates a buffer of
    TargetBuffer
      ( segment::Segment & segment
      , std::size_t size );

    TargetBuffer
      ( void * const ptr
      , segment::Segment & segment
      , std::size_t size );

    TargetBuffer
      ( segment::Segment & segment
      , std::size_t size
      , segment::Segment
          ::Notification notification );

    TargetBuffer
      ( void * const ptr
      , segment::Segment & segment
      , std::size_t size
      , segment::Segment
          ::Notification notification );

    // bilateral function
    // needs to be invoked by the correspondent
    // WriteTargetBuffer having the same size
    Endpoint::ConnectHandle
    connectToRemoteSource
      ( Context & context
      , group::Rank & rank
      , Tag & tag );

    bool
    waitForCompletion
      ();

    bool
    checkForCompletion
      ();

    void
    ackTransfer
      (Context & context);
};

} // namespace write
} // namespace singlesided

} /* namespace gaspi */

#endif /* GASPICXX_HPP_ */
