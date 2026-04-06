using MedicalAssistant.Domain.Entities;

namespace MedicalAssistant.Domain.Contracts
{
    public interface IUnitOfWork : IAsyncDisposable, IDisposable
    {
        IPatientRepository Patients { get; }

        IAppointmentRepository Appointments { get; }

        IDoctorRepository Doctors { get; }

        IReviewRepository Reviews { get; }

        IAdminRepository Admins { get; }

        IGenericRepository<TEntity> Repository<TEntity>() where TEntity : BaseEntity;

        Task<int> SaveChangesAsync();

        Task BeginTransactionAsync();
        Task CommitTransactionAsync();
        Task RollbackTransactionAsync();
    }
}