using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities;
using MedicalAssistant.Persistance.Data.DbContexts;
using Microsoft.EntityFrameworkCore.Storage;
using System.Collections;

namespace MedicalAssistant.Persistance.Repositories;

public class UnitOfWork : IUnitOfWork
{
    private readonly MedicalAssistantDbContext _context;
    private IDbContextTransaction? _transaction;
    private bool _disposed;

    // قاموس لتخزين الـ Repositories المنشأة لضمان عدم تكرارها (Flyweight Pattern)
    private Hashtable? _repositories;

    private IPatientRepository? _patients;
    private IAppointmentRepository? _appointments;
    private IDoctorRepository? _doctors;

    public UnitOfWork(MedicalAssistantDbContext context)
    {
        _context = context;
    }

    // --- تحقق الخصائص (Properties Implementation) ---
    public IPatientRepository Patients => _patients ??= new PatientRepository(_context);
    public IAppointmentRepository Appointments => _appointments ??= new AppointmentRepository(_context);
    public IDoctorRepository Doctors => _doctors ??= new DoctorRepository(_context);

    // --- تنفيذ الـ Generic Repository الوصول العام ---
    public IGenericRepository<TEntity> Repository<TEntity>() where TEntity : BaseEntity
    {
        _repositories ??= new Hashtable();

        var type = typeof(TEntity).Name;

        if (!_repositories.ContainsKey(type))
        {
            var repositoryType = typeof(GenericRepository<>);
            var repositoryInstance = Activator.CreateInstance(repositoryType.MakeGenericType(typeof(TEntity)), _context);
            _repositories.Add(type, repositoryInstance);
        }

        return (IGenericRepository<TEntity>)_repositories[type]!;
    }

    public async Task<int> SaveChangesAsync()
    {
        return await _context.SaveChangesAsync();
    }

    // --- إدارة العمليات (Transaction Management) ---
    public async Task BeginTransactionAsync()
    {
        _transaction = await _context.Database.BeginTransactionAsync();
    }

    public async Task CommitTransactionAsync()
    {
        try
        {
            await _context.SaveChangesAsync();
            if (_transaction != null) await _transaction.CommitAsync();
        }
        catch
        {
            await RollbackTransactionAsync();
            throw;
        }
        finally
        {
            if (_transaction != null)
            {
                await _transaction.DisposeAsync();
                _transaction = null;
            }
        }
    }

    public async Task RollbackTransactionAsync()
    {
        if (_transaction != null)
        {
            await _transaction.RollbackAsync();
            await _transaction.DisposeAsync();
            _transaction = null;
        }
    }

    // --- التخلص من الكائنات (Disposal) ---
    public async ValueTask DisposeAsync()
    {
        await _context.DisposeAsync();
        if (_transaction != null) await _transaction.DisposeAsync();
    }

    public void Dispose()
    {
        _context.Dispose();
        _transaction?.Dispose();
        GC.SuppressFinalize(this);
    }
}